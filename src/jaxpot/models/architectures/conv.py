from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from flax import nnx

from jaxpot.models.base import ComposablePolicyValueModel, ModelOutput
from jaxpot.models.utils import orthogonal_init


class ConvModel(ComposablePolicyValueModel):
    """Universal CNN model for board games with spatial observations.

    Architecture:
        - 2D convolutions (3x3, SAME padding) over board planes
        - Flatten and MLP hidden layers
        - Separate policy and value heads

    Parameters
    ----------
    rngs
        Random number generators for submodules.
    action_dim
        Number of policy logits to output.
    obs_shape
        Shape of the observation (H, W, C). Used to derive spatial dims
        and input channels.
    conv_channels
        List of channel sizes for convolutional layers.
    hidden_dims
        Sizes of hidden MLP layers after flattening.
    activation
        Activation function.
    """

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        action_dim: int | tuple[int, ...],
        obs_shape: int | tuple[int, ...],
        conv_channels: list[int] | None = None,
        hidden_dims: int | list[int] = 128,
        activation: Callable[[jnp.ndarray], jnp.ndarray] = nnx.relu,
        **kwargs,
    ):
        super().__init__(obs_shape=obs_shape, action_dim=action_dim, **kwargs)
        if len(self.obs_shape) != 3:
            raise ValueError(f"ConvModel requires 3D obs_shape (H, W, C), got {self.obs_shape}")
        if conv_channels is None:
            conv_channels = [64, 64]

        self.hidden_dims = [hidden_dims] if isinstance(hidden_dims, int) else list(hidden_dims)
        self.activation = activation

        # Convolutional layers
        conv_layers = []
        in_channels = self.obs_shape[2]
        for out_channels in conv_channels:
            conv_layers.append(
                nnx.Conv(
                    in_features=in_channels,
                    out_features=out_channels,
                    kernel_size=(3, 3),
                    padding="SAME",
                    rngs=rngs,
                )
            )
            in_channels = out_channels
        self.conv_layers = nnx.List(conv_layers)

        # MLP layers after flatten
        flat_size = self.obs_shape[0] * self.obs_shape[1] * conv_channels[-1]
        mlp_layers = []
        in_features = flat_size
        for out_features in self.hidden_dims:
            mlp_layers.append(
                nnx.Linear(
                    in_features=in_features,
                    out_features=out_features,
                    rngs=rngs,
                )
            )
            in_features = out_features
        self.mlp_layers = nnx.List(mlp_layers)

        head_in_features = self.hidden_dims[-1] if self.hidden_dims else flat_size
        self.policy_head = nnx.Linear(
            in_features=head_in_features,
            out_features=self.action_dim,
            kernel_init=orthogonal_init(0.01),
            rngs=rngs,
        )
        self.value_head = nnx.Linear(
            in_features=head_in_features,
            out_features=1,
            kernel_init=orthogonal_init(1.0),
            rngs=rngs,
        )

    def encode(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Encode spatial observation into a feature vector.

        Parameters
        ----------
        obs
            Batched observation ``[B, H, W, C]``.

        Returns
        -------
        jnp.ndarray
            Feature vector ``[B, D]``.
        """
        x = obs
        for conv in self.conv_layers:
            x = conv(x)
            x = self.activation(x)
        x = x.reshape(x.shape[0], -1)
        for layer in self.mlp_layers:
            x = layer(x)
            x = self.activation(x)
        return x

    def decode(self, core_output: jnp.ndarray) -> ModelOutput:
        """Compute policy logits and value from feature vector."""
        return ModelOutput(
            value=self.value_head(core_output),
            policy_logits=self.policy_head(core_output),
        )
