from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from flax import nnx

from jaxpot.models.base import ComposablePolicyValueModel, ModelOutput
from jaxpot.models.utils import orthogonal_init


class ResidualMLPBlock(nnx.Module):
    """Residual block for flat (MLP) architectures.

    LayerNorm -> Linear -> activation -> Linear + skip connection.
    """

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        hidden_dim: int,
        activation: Callable[[jnp.ndarray], jnp.ndarray] = nnx.relu,
    ):
        self.ln = nnx.LayerNorm(num_features=hidden_dim, rngs=rngs)
        self.linear1 = nnx.Linear(
            in_features=hidden_dim,
            out_features=hidden_dim,
            rngs=rngs,
        )
        self.linear2 = nnx.Linear(
            in_features=hidden_dim,
            out_features=hidden_dim,
            rngs=rngs,
        )
        self.activation = activation

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        x = self.ln(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x + residual


class ResidualMLPModel(ComposablePolicyValueModel):
    """Residual MLP model for flat observations.

    Architecture:
        - Input projection: input_dim -> hidden_dim
        - N residual blocks (LayerNorm -> Linear -> activation -> Linear + skip)
        - Final LayerNorm on trunk output
        - Policy head: Linear(hidden_dim, action_dim)
        - Value head: Linear -> activation -> LayerNorm -> Linear(1)

    Parameters
    ----------
    rngs
        Random number generators for submodules.
    action_dim
        Number of policy logits to output.
    obs_shape
        Shape of the observation (will be flattened).
    hidden_dim
        Hidden dimension for residual blocks.
    num_blocks
        Number of residual blocks.
    value_hidden_dim
        Hidden dimension for value head MLP.
    activation
        Activation function.
    """

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        action_dim: int | tuple[int, ...],
        obs_shape: int | tuple[int, ...],
        hidden_dim: int = 256,
        num_blocks: int = 4,
        value_hidden_dim: int = 128,
        activation: Callable[[jnp.ndarray], jnp.ndarray] = nnx.relu,
        **kwargs,
    ):
        super().__init__(obs_shape=obs_shape, action_dim=action_dim, **kwargs)
        action_dim = self.action_dim
        self.hidden_dim = hidden_dim
        self.activation = activation

        # Input projection
        self.input_proj = nnx.Linear(
            in_features=self.input_dim,
            out_features=hidden_dim,
            rngs=rngs,
        )

        # Residual tower
        self.res_blocks = nnx.List(
            [
                ResidualMLPBlock(rngs=rngs, hidden_dim=hidden_dim, activation=activation)
                for _ in range(num_blocks)
            ]
        )

        # Final LayerNorm on trunk
        self.trunk_ln = nnx.LayerNorm(num_features=hidden_dim, rngs=rngs)

        # Policy head
        self.policy_head = nnx.Linear(
            in_features=hidden_dim,
            out_features=self.action_dim,
            kernel_init=orthogonal_init(0.01),
            rngs=rngs,
        )

        # Value head
        self.value_fc1 = nnx.Linear(
            in_features=hidden_dim,
            out_features=value_hidden_dim,
            rngs=rngs,
        )
        self.value_ln = nnx.LayerNorm(num_features=value_hidden_dim, rngs=rngs)
        self.value_fc2 = nnx.Linear(
            in_features=value_hidden_dim,
            out_features=1,
            kernel_init=orthogonal_init(1.0),
            rngs=rngs,
        )

    def encode(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Encode observation into a feature vector.

        Parameters
        ----------
        obs
            Batched observation ``[B, *obs_shape]``.

        Returns
        -------
        jnp.ndarray
            Feature vector ``[B, hidden_dim]``.
        """
        x = obs
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)

        x = self.input_proj(x)
        x = self.activation(x)

        for block in self.res_blocks:
            x = block(x)

        return self.trunk_ln(x)

    def decode(self, core_output: jnp.ndarray) -> ModelOutput:
        """Compute policy logits and value from feature vector."""
        policy_logits = self.policy_head(core_output)

        value = self.value_fc1(core_output)
        value = self.activation(value)
        value = self.value_ln(value)
        value = self.value_fc2(value)

        return ModelOutput(value=value, policy_logits=policy_logits)
