"""
ResNet + LSTM model for board game environments.

Architecture:
- ResNet encoder (same as ResNetModel: stem + residual blocks + global avg pool)
- LSTM core processing the feature vector with recurrent state
- Separate policy and value heads from LSTM output
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
from flax import nnx

from jaxpot.models.base import ComposablePolicyValueModel, ModelOutput
from jaxpot.models.blocks import ConvResidualBlock, LSTMCore
from jaxpot.models.utils import he_normal_init, orthogonal_init


class ResNetLSTMModel(ComposablePolicyValueModel):
    """ResNet encoder + LSTM core + policy/value heads.

    Architecture::

        obs [B, H, W, C]
          -> ResNet encode: stem + residual blocks + global avg pool -> [B, num_filters]
          -> LSTM core: OptimizedLSTMCell(num_filters, lstm_hidden_size) -> [B, lstm_hidden_size]
          -> decode: policy_head(lstm_hidden_size -> action_dim),
                     value_head(lstm_hidden_size -> 1)

    Parameters
    ----------
    rngs
        Random number generators.
    action_dim
        Shape of the action space.
    obs_shape
        Shape of a single observation ``(H, W, C)``.
    num_filters
        Number of convolutional feature channels.
    num_blocks
        Number of convolutional residual blocks.
    lstm_hidden_size
        LSTM hidden state size.
    num_lstm_layers
        Number of stacked LSTM layers.
    activation
        Activation function for residual blocks.
    dropout
        Dropout probability in residual blocks.
    """

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        action_dim: tuple[int, ...],
        obs_shape: tuple[int, ...],
        num_filters: int = 128,
        num_blocks: int = 6,
        lstm_hidden_size: int = 256,
        num_lstm_layers: int = 1,
        activation: Callable[[jnp.ndarray], jnp.ndarray] = nnx.relu,
        dropout: float = 0.0,
    ):
        super().__init__(obs_shape=obs_shape, action_dim=action_dim)

        in_channels = int(self.obs_shape[-1])
        flat_action_dim = self.action_dim

        self.activation = activation

        # ResNet encoder
        self.stem = nnx.Conv(
            in_features=in_channels,
            out_features=num_filters,
            kernel_size=(3, 3),
            padding="SAME",
            kernel_init=he_normal_init(),
            rngs=rngs,
        )
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

        # LSTM core
        self.lstm_core = LSTMCore(
            rngs=rngs,
            input_size=num_filters,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
        )

        # Policy and value heads
        self.policy_head = nnx.Linear(
            in_features=lstm_hidden_size,
            out_features=flat_action_dim,
            kernel_init=orthogonal_init(0.01),
            rngs=rngs,
        )
        self.value_head = nnx.Linear(
            in_features=lstm_hidden_size,
            out_features=1,
            kernel_init=orthogonal_init(1.0),
            rngs=rngs,
        )

    def encode(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Encode spatial observation into a feature vector.

        Parameters
        ----------
        obs
            Board observation of shape ``(B, H, W, C)``.

        Returns
        -------
        jnp.ndarray
            Feature vector of shape ``(B, num_filters)``.
        """
        x = self.stem(obs)
        x = self.activation(x)
        for block in self.blocks:
            x = block(x)
        return jnp.mean(x, axis=(1, 2))

    def core(
        self, features: jnp.ndarray, state: jnp.ndarray | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Process features through the LSTM core.

        ``state`` is a single opaque tensor of shape ``[B, 2, L, H]`` packed
        by :class:`LSTMCore`. The framework treats it as opaque.
        """
        return self.lstm_core(features, state)

    def decode(self, core_output: jnp.ndarray) -> ModelOutput:
        """Compute policy logits and value from LSTM output.

        Parameters
        ----------
        core_output
            LSTM output ``[B, lstm_hidden_size]``.

        Returns
        -------
        ModelOutput
        """
        return ModelOutput(
            value=self.value_head(core_output),
            policy_logits=self.policy_head(core_output),
        )

    @property
    def hidden_shape(self) -> tuple[int, int, int]:
        return self.lstm_core.state_shape

    def init_state(self, batch_size: int) -> jnp.ndarray:
        """Initialize LSTM hidden state with zeros, shape ``[B, 2, L, H]``."""
        return self.lstm_core.init_state(batch_size)

    @property
    def is_recurrent(self) -> bool:
        return True
