"""
AlphaZero-style CNN (AZNet) for pgx board games.

Same architecture and defaults as ``pgx/examples/alphazero/network.py``:
3x3 stem, ResNet v1/v2 residual blocks with BatchNorm, 1x1 conv policy and value
heads, two-layer value MLP, tanh on value.

Observations are (H, W, C) in channels-last order. Default width/depth match the
previous ``AZNetModel`` wrapper (``num_channels=128``, ``num_blocks=6``,
``resnet_v2=True``).

``num_filters`` is accepted as an alias for ``num_channels`` (backward compatibility).
"""

from __future__ import annotations

import jax.nn as jnn
import jax.numpy as jnp
from flax import nnx
from loguru import logger

from jaxpot.models.base import ModelOutput, PolicyValueModel
from jaxpot.models.utils import he_normal_init, orthogonal_init


class _BatchNorm(nnx.Module):
    """Batch normalization using only batch statistics — no running-average state.

    All variables are ``nnx.Param`` (scale and bias), so the module is fully
    compatible with trainers that treat every model variable as trainable.

    This matches Haiku's ``BatchNorm(True, True, 0.9)`` called with
    ``is_training=True``, which is the mode used by the pgx AlphaZero reference
    implementation during training.
    """

    def __init__(
        self,
        num_features: int,
        *,
        epsilon: float = 1e-5,
        rngs: nnx.Rngs,
        **_ignored,
    ) -> None:
        self.scale = nnx.Param(jnp.ones((num_features,)))
        self.bias = nnx.Param(jnp.zeros((num_features,)))
        self.epsilon = epsilon

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Reduce over all axes except the last (channel) dimension
        axis = tuple(range(x.ndim - 1))
        mean = jnp.mean(x, axis=axis, keepdims=True)
        var = jnp.var(x, axis=axis, keepdims=True)
        x_norm = (x - mean) / jnp.sqrt(var + self.epsilon)
        return self.scale.value * x_norm + self.bias.value


class _BlockV1(nnx.Module):
    """ResNet v1-style block: conv-BN-ReLU-conv-BN, then ReLU(residual + x)."""

    def __init__(
        self,
        num_channels: int,
        *,
        bn_momentum: float,
        rngs: nnx.Rngs,
    ) -> None:
        self.conv1 = nnx.Conv(
            in_features=num_channels,
            out_features=num_channels,
            kernel_size=(3, 3),
            padding="SAME",
            kernel_init=he_normal_init(),
            rngs=rngs,
        )
        self.bn1 = _BatchNorm(num_channels, rngs=rngs)
        self.conv2 = nnx.Conv(
            in_features=num_channels,
            out_features=num_channels,
            kernel_size=(3, 3),
            padding="SAME",
            kernel_init=he_normal_init(),
            rngs=rngs,
        )
        self.bn2 = _BatchNorm(num_channels, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        i = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = jnn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return jnn.relu(x + i)


class _BlockV2(nnx.Module):
    """ResNet v2-style (pre-activation) block: BN-ReLU-conv-BN-ReLU-conv + residual."""

    def __init__(
        self,
        num_channels: int,
        *,
        bn_momentum: float,
        rngs: nnx.Rngs,
    ) -> None:
        self.bn1 = _BatchNorm(num_channels, rngs=rngs)
        self.conv1 = nnx.Conv(
            in_features=num_channels,
            out_features=num_channels,
            kernel_size=(3, 3),
            padding="SAME",
            kernel_init=he_normal_init(),
            rngs=rngs,
        )
        self.bn2 = _BatchNorm(num_channels, rngs=rngs)
        self.conv2 = nnx.Conv(
            in_features=num_channels,
            out_features=num_channels,
            kernel_size=(3, 3),
            padding="SAME",
            kernel_init=he_normal_init(),
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        i = x
        x = self.bn1(x)
        x = jnn.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = jnn.relu(x)
        x = self.conv2(x)
        return x + i


class AZNetModel(PolicyValueModel):
    """pgx ``AZNet`` architecture in Flax NNX; implements :class:`PolicyValueModel` fully."""

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        action_dim: tuple[int, ...],
        obs_shape: tuple[int, ...],
        num_channels: int | None = None,
        num_filters: int | None = None,
        num_blocks: int = 6,
        resnet_v2: bool = True,
        bn_momentum: float = 0.9,
    ) -> None:
        super().__init__(obs_shape=obs_shape, action_dim=action_dim)

        if len(self.obs_shape) != 3:
            raise ValueError(f"AZNetModel requires 3D obs_shape (H, W, C), got {self.obs_shape}")

        if num_channels is not None and num_filters is not None and num_channels != num_filters:
            raise ValueError("Pass at most one of num_channels and num_filters")
        if num_channels is not None:
            width = int(num_channels)
        elif num_filters is not None:
            width = int(num_filters)
        else:
            width = 128

        in_ch = int(self.obs_shape[-1])
        self._h = int(self.obs_shape[0])
        self._w = int(self.obs_shape[1])
        self._num_channels = width
        self._resnet_v2 = resnet_v2
        self._spatial_flat = self._h * self._w

        self.stem = nnx.Conv(
            in_features=in_ch,
            out_features=width,
            kernel_size=(3, 3),
            padding="SAME",
            kernel_init=he_normal_init(),
            rngs=rngs,
        )

        if not resnet_v2:
            self.stem_bn = _BatchNorm(width, rngs=rngs)
        else:
            self.stem_bn = None

        block_cls = _BlockV2 if resnet_v2 else _BlockV1
        self.blocks = nnx.List(
            [block_cls(width, bn_momentum=bn_momentum, rngs=rngs) for _ in range(num_blocks)]
        )

        if resnet_v2:
            self.tower_final_bn = _BatchNorm(width, rngs=rngs)
        else:
            self.tower_final_bn = None

        self.policy_conv = nnx.Conv(
            in_features=width,
            out_features=2,
            kernel_size=(1, 1),
            padding="SAME",
            kernel_init=he_normal_init(),
            rngs=rngs,
        )
        self.policy_bn = _BatchNorm(2, rngs=rngs)
        policy_flat_in = self._spatial_flat * 2
        self.policy_head = nnx.Linear(
            in_features=policy_flat_in,
            out_features=self.action_dim,
            kernel_init=orthogonal_init(0.01),
            rngs=rngs,
        )

        self.value_conv = nnx.Conv(
            in_features=width,
            out_features=1,
            kernel_size=(1, 1),
            padding="SAME",
            kernel_init=he_normal_init(),
            rngs=rngs,
        )
        self.value_bn = _BatchNorm(1, rngs=rngs)
        value_flat_in = self._spatial_flat
        self.value_fc1 = nnx.Linear(
            in_features=value_flat_in,
            out_features=width,
            kernel_init=orthogonal_init(1.0),
            rngs=rngs,
        )
        self.value_head = nnx.Linear(
            in_features=width,
            out_features=1,
            kernel_init=orthogonal_init(1.0),
            rngs=rngs,
        )

    @property
    def features_dim(self) -> int:
        """Flattened spatial trunk size (for auxiliary heads such as game progress)."""
        return self._spatial_flat * self._num_channels

    def _trunk(self, obs_bhwc: jnp.ndarray) -> jnp.ndarray:
        x = obs_bhwc.astype(jnp.float32)
        x = self.stem(x)

        if not self._resnet_v2:
            assert self.stem_bn is not None
            x = self.stem_bn(x)
            x = jnn.relu(x)

        for block in self.blocks:
            x = block(x)

        if self._resnet_v2:
            assert self.tower_final_bn is not None
            x = self.tower_final_bn(x)
            x = jnn.relu(x)

        return x

    def _normalize_obs(self, obs: jnp.ndarray) -> tuple[jnp.ndarray, bool] | None:
        """Returns (batched_obs, squeeze_output) or None if empty batch (caller handles)."""
        squeeze_output = False

        if obs.ndim == 3:
            obs = obs[None, ...]
            squeeze_output = True
        elif obs.ndim == 4:
            if obs.shape[0] == 0:
                return None
        else:
            raise ValueError(
                f"Expected obs to have 3 or 4 dimensions (H,W,C) or (B,H,W,C), "
                f"got {obs.ndim} with shape {obs.shape}"
            )

        return obs, squeeze_output

    def features(self, obs: jnp.ndarray) -> tuple[jnp.ndarray, bool]:
        normalized = self._normalize_obs(obs)
        if normalized is None:
            raise ValueError("features() does not support empty batch; use __call__ instead")

        obs_b, squeeze_output = normalized
        trunk = self._trunk(obs_b)
        h = trunk.reshape((trunk.shape[0], -1))
        return h, squeeze_output

    def heads(self, h: jnp.ndarray, squeeze_output: bool) -> ModelOutput:
        x = h.reshape((h.shape[0], self._h, self._w, self._num_channels))

        p = self.policy_conv(x)
        p = self.policy_bn(p)
        p = jnn.relu(p)
        p = p.reshape((p.shape[0], -1))
        policy_logits = self.policy_head(p)

        v = self.value_conv(x)
        v = self.value_bn(v)
        v = jnn.relu(v)
        v = v.reshape((v.shape[0], -1))
        v = self.value_fc1(v)
        v = jnn.relu(v)
        v = self.value_head(v)
        value = jnp.tanh(v)

        if squeeze_output:
            value = value.squeeze(0)
            policy_logits = policy_logits.squeeze(0)

        return ModelOutput(value=value, policy_logits=policy_logits)

    def __call__(
        self, obs: jnp.ndarray, hidden_state: jnp.ndarray | None = None
    ) -> ModelOutput:
        # AZNet is non-recurrent; the optional hidden_state is accepted for
        # interface uniformity and ignored.
        del hidden_state
        normalized = self._normalize_obs(obs)
        if normalized is None:
            logger.warning("Empty batch case - returning early with empty outputs")
            flat_action_dim = int(jnp.prod(jnp.array(self.action_dim)).item())
            return ModelOutput(
                value=jnp.empty((0, 1), dtype=obs.dtype),
                policy_logits=jnp.empty((0, flat_action_dim), dtype=obs.dtype),
            )

        obs_b, squeeze_output = normalized
        trunk = self._trunk(obs_b)
        h = trunk.reshape((trunk.shape[0], -1))
        return self.heads(h, squeeze_output)
