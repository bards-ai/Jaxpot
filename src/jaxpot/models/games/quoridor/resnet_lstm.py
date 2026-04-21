"""
ResNet + LSTM model for Quoridor with scalar feature support.

Extends ResNetLSTMModel to handle Quoridor's flat observation layout:
  [spatial_flat (324,), scalar (2,)] = (326,)

The spatial part is reshaped to (9, 9, 4) for the ResNet encoder.
Scalar features (walls remaining) are concatenated after encoding,
before the LSTM core.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from flax import nnx

from jaxpot.env.quoridor.observation import (
    BOARD_SIZE,
    NUM_SCALAR_FEATURES,
    NUM_SPATIAL_CHANNELS,
    SPATIAL_SIZE,
)
from jaxpot.models.architectures.resnet_lstm import ResNetLSTMModel
from jaxpot.models.base import ModelOutput
from jaxpot.models.blocks import ConvResidualBlock, LSTMCore
from jaxpot.models.utils import he_normal_init, orthogonal_init


class QuoridorResNetLSTMModel(ResNetLSTMModel):
    """ResNet + LSTM for Quoridor with scalar features concatenated before LSTM.

    Architecture::

        obs [B, 326]
          -> split: spatial [B, 9, 9, 4], scalar [B, 2]
          -> ResNet encode: spatial -> [B, num_filters]
          -> concat([B, num_filters], [B, 2]) -> [B, num_filters + 2]
          -> LSTM core -> [B, lstm_hidden_size]
          -> policy_head, value_head
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
        # Skip ResNetLSTMModel.__init__ — we build everything ourselves
        # because LSTM input_size differs (num_filters + NUM_SCALAR_FEATURES)
        super(ResNetLSTMModel, self).__init__(obs_shape=obs_shape, action_dim=action_dim)

        self.activation = activation

        # ResNet encoder (spatial only, 4 input channels)
        self.stem = nnx.Conv(
            in_features=NUM_SPATIAL_CHANNELS,
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

        # LSTM core: input is resnet features + scalar features
        lstm_input_size = num_filters + NUM_SCALAR_FEATURES
        self.lstm_core = LSTMCore(
            rngs=rngs,
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
        )

        flat_action_dim = self.action_dim

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
        """Split flat obs into spatial + scalar, encode spatial with ResNet,
        then concat scalar features.

        Parameters
        ----------
        obs
            Flat observation ``[B, 326]``.

        Returns
        -------
        jnp.ndarray
            Feature vector ``[B, num_filters + 2]``.
        """
        # Split flat obs
        spatial_flat = obs[..., :SPATIAL_SIZE]
        scalar = obs[..., SPATIAL_SIZE:]

        # Reshape spatial to (B, 9, 9, 4)
        batch_shape = spatial_flat.shape[:-1]
        spatial = spatial_flat.reshape(*batch_shape, BOARD_SIZE, BOARD_SIZE, NUM_SPATIAL_CHANNELS)

        # ResNet encode
        x = self.stem(spatial)
        x = self.activation(x)
        for block in self.blocks:
            x = block(x)
        spatial_features = jnp.mean(x, axis=(-3, -2))  # global avg pool -> [B, num_filters]

        # Concat scalar features
        return jnp.concatenate([spatial_features, scalar], axis=-1)
