"""
Transformer encoder + LSTM model for board game environments.

Architecture:
- Vision Transformer encoder: flatten spatial dims to tokens, project, add positional
  embeddings, pass through TransformerBlock layers, pool via AttentionPooling
- LSTM core processing the pooled feature vector with recurrent state
- Separate policy and value heads from LSTM output
"""

from __future__ import annotations

import math
from typing import Any

import jax.numpy as jnp
from flax import nnx

from jaxpot.models.base import ComposablePolicyValueModel, ModelOutput
from jaxpot.models.blocks import AttentionPooling, LSTMCore, TransformerBlock
from jaxpot.models.utils import orthogonal_init


class TransformerLSTMModel(ComposablePolicyValueModel):
    """Transformer encoder + LSTM core + policy/value heads.

    Treats each board position as a token, applies self-attention, then
    pools into a single feature vector for the LSTM.

    Architecture::

        obs [B, H, W, C]
          -> reshape to [B, H*W, C]
          -> Linear(C, embed_dim) + learned positional embedding
          -> N x TransformerBlock
          -> AttentionPooling -> [B, embed_dim]
          -> LSTM core -> [B, lstm_hidden_size]
          -> policy_head, value_head

    Parameters
    ----------
    rngs
        Random number generators.
    action_dim
        Shape of the action space.
    obs_shape
        Shape of a single observation ``(H, W, C)``.
    embed_dim
        Transformer embedding dimension.
    num_heads
        Number of attention heads.
    num_blocks
        Number of transformer encoder blocks.
    expansion
        MLP expansion factor in transformer blocks.
    dropout
        Dropout probability.
    lstm_hidden_size
        LSTM hidden state size.
    num_lstm_layers
        Number of stacked LSTM layers.
    """

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        action_dim: tuple[int, ...],
        obs_shape: tuple[int, ...],
        embed_dim: int = 128,
        num_heads: int = 4,
        num_blocks: int = 4,
        expansion: int = 4,
        dropout: float = 0.0,
        lstm_hidden_size: int = 256,
        num_lstm_layers: int = 1,
    ):
        super().__init__(obs_shape=obs_shape, action_dim=action_dim)

        in_channels = int(self.obs_shape[-1])
        flat_action_dim = self.action_dim
        num_tokens = math.prod(self.obs_shape[:-1])

        # Token projection
        self.token_proj = nnx.Linear(
            in_features=in_channels,
            out_features=embed_dim,
            rngs=rngs,
        )

        # Learned positional embedding
        self.pos_embed = nnx.Param(
            nnx.initializers.normal(stddev=0.02)(rngs.params(), (1, num_tokens, embed_dim))
        )

        # Transformer encoder blocks
        self.blocks = nnx.List(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    expansion=expansion,
                    rngs=rngs,
                )
                for _ in range(num_blocks)
            ]
        )

        # Attention pooling
        self.pool = AttentionPooling(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            rngs=rngs,
        )

        # LSTM core
        self.lstm_core = LSTMCore(
            rngs=rngs,
            input_size=embed_dim,
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
            Feature vector of shape ``(B, embed_dim)``.
        """
        B = obs.shape[0]
        # Flatten spatial dims to token sequence: [B, H, W, C] -> [B, H*W, C]
        x = obs.reshape(B, -1, obs.shape[-1])
        # Project to embedding dim and add positional embedding
        x = self.token_proj(x) + self.pos_embed.value
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        # Attention pooling -> [B, embed_dim]
        return self.pool(x)

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
