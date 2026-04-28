from __future__ import annotations

import jax.numpy as jnp
from flax import nnx


class LSTMCore(nnx.Module):
    """Reusable LSTM core that wraps stacked OptimizedLSTMCell layers.

    The recurrent state is exposed externally as a single batch-first tensor
    of shape ``[B, 2, num_layers, hidden_size]`` where index 0 along axis 1
    is the LSTM hidden state ``h`` and index 1 is the cell state ``c``. This
    keeps the model's recurrent contract to the rest of the framework as a
    single opaque tensor — only the LSTM internals know about ``(h, c)``.
    """

    def __init__(self, *, rngs: nnx.Rngs, input_size: int, hidden_size: int, num_layers: int = 1):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nnx.List(
            [
                nnx.OptimizedLSTMCell(
                    in_features=input_size if i == 0 else hidden_size,
                    hidden_features=hidden_size,
                    rngs=rngs,
                )
                for i in range(num_layers)
            ]
        )

    @property
    def state_shape(self) -> tuple[int, int, int]:
        """Per-sample opaque state shape: ``(2, num_layers, hidden_size)``."""
        return (2, self.num_layers, self.hidden_size)

    def init_state(self, batch_size: int) -> jnp.ndarray:
        """Return zero recurrent state of shape ``[B, 2, num_layers, hidden_size]``."""
        return jnp.zeros((batch_size, *self.state_shape))

    def __call__(
        self, features: jnp.ndarray, state: jnp.ndarray | None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if state is None:
            state = self.init_state(features.shape[0])
        # state: [B, 2, L, H] -> per-layer (h, c) each [B, H]
        h_stack = state[:, 0]  # [B, L, H]
        c_stack = state[:, 1]  # [B, L, H]
        new_h_layers = []
        new_c_layers = []
        x = features
        for i, cell in enumerate(self.cells):
            carry = (h_stack[:, i], c_stack[:, i])
            (new_h, new_c), y = cell(carry, x)
            new_h_layers.append(new_h)
            new_c_layers.append(new_c)
            x = y
        new_h_stack = jnp.stack(new_h_layers, axis=1)  # [B, L, H]
        new_c_stack = jnp.stack(new_c_layers, axis=1)  # [B, L, H]
        new_state = jnp.stack([new_h_stack, new_c_stack], axis=1)  # [B, 2, L, H]
        return x, new_state
