"""
Convolutional ResNet model for pgx board game environments.

Architecture:
- Conv2D stem projecting input channels to num_filters
- N x ConvResidualBlock (pre-activation, identity skip connections)
- Global average pooling to collapse spatial dimensions
- Separate policy and value heads

Designed for spatial observations of shape (H, W, C) such as those produced
by pgx go and chess environments.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import jax.nn as jnn
import jax.numpy as jnp
from flax import nnx

from jaxpot.models.base import ComposablePolicyValueModel, ModelOutput
from jaxpot.models.blocks import ConvResidualBlock
from jaxpot.models.utils import he_normal_init, orthogonal_init


class ResNetModel(ComposablePolicyValueModel):
    """Convolutional ResNet model for pgx board game environments.

    Architecture:
        - Conv2D stem (input channels -> num_filters)
        - N x ConvResidualBlock with identity skip connections
        - Global average pooling over spatial dimensions
        - Separate policy and value heads

    Parameters
    ----------
    rngs
        Random number generators for submodules.
    action_dim
        Shape of the action space.
    obs_shape
        Shape of a single observation, expected to be (H, W, C).
    num_filters
        Number of convolutional feature channels used throughout the network.
    num_blocks
        Number of convolutional residual blocks.
    activation
        Activation function applied inside each residual block.
    dropout
        Dropout probability applied inside each residual block.
    """

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        action_dim: tuple[int, ...],
        obs_shape: tuple[int, ...],
        num_filters: int = 128,
        num_blocks: int = 6,
        activation: Callable[[jnp.ndarray], jnp.ndarray] = jnn.relu,
        dropout: float = 0.0,
    ):
        super().__init__(obs_shape=obs_shape, action_dim=action_dim)

        in_channels = int(self.obs_shape[-1])
        flat_action_dim = self.action_dim

        self.activation = activation

        # Stem: project input channels to num_filters
        self.stem = nnx.Conv(
            in_features=in_channels,
            out_features=num_filters,
            kernel_size=(3, 3),
            padding="SAME",
            kernel_init=he_normal_init(),
            rngs=rngs,
        )

        # Residual tower
        self.blocks = nnx.List(
            [
                ConvResidualBlock(
                    num_filters=num_filters,
                    dropout=dropout,
                    activation=activation,
                    rngs=rngs,
                )
                for _ in range(num_blocks)
            ]
        )

        # Policy head
        self.policy_head = nnx.Linear(
            in_features=num_filters,
            out_features=flat_action_dim,
            kernel_init=orthogonal_init(0.01),
            rngs=rngs,
        )

        # Value head
        self.value_head = nnx.Linear(
            in_features=num_filters,
            out_features=1,
            kernel_init=orthogonal_init(1.0),
            rngs=rngs,
        )

    def encode(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Encode spatial observation into a feature vector.

        Parameters
        ----------
        obs
            Board observation of shape (B, H, W, C).

        Returns
        -------
        jnp.ndarray
            Feature vector of shape (B, num_filters).
        """
        x = self.stem(obs)
        x = self.activation(x)
        for block in self.blocks:
            x = block(x)
        return jnp.mean(x, axis=(1, 2))

    def decode(self, core_output: jnp.ndarray) -> ModelOutput:
        """Compute policy logits and value from feature vector.

        Parameters
        ----------
        core_output
            Feature vector of shape (B, D).

        Returns
        -------
        ModelOutput
        """
        return ModelOutput(
            value=self.value_head(core_output),
            policy_logits=self.policy_head(core_output),
        )

