"""
Minimal LSTM Model for pgx-style observations.

Architecture:
- Flatten observation features per timestep
- LSTM encoder over sequence
- Optional post-LSTM MLP projection
- Outputs: value and policy logits

Supports both non-sequential and sequential observations:
- Non-sequential: (B, H, W, C), (H, W, C), (B, F), (F,)
- Sequential: (B, T, H, W, C), (B, T, F)
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx

from jaxpot.models.base import ComposablePolicyValueModel, ModelOutput
from jaxpot.models.utils import orthogonal_init


class LSTMModel(ComposablePolicyValueModel):
    """LSTM model.

    Parameters
    ----------
    rngs
        Random number generators for submodules.
    action_dim
        Shape of the action logits output.
    obs_shape
        Shape of a single observation.
    hidden_size
        Hidden size of each LSTM layer.
    num_lstm_layers
        Number of stacked LSTM layers.
    bidirectional
        If True, run forward and backward LSTM and concatenate outputs.
    sequence_pooling
        How to pool sequence outputs: "last", "mean", or "max".
    head_hidden_dims
        Optional hidden MLP layers after sequence encoder and before heads.
    activation
        Activation function used in post-LSTM MLP.
    """

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        action_dim: int | tuple[int, ...],
        obs_shape: int | tuple[int, ...],
        hidden_size: int = 128,
        num_lstm_layers: int = 1,
        bidirectional: bool = False,
        sequence_pooling: str = "last",
        head_hidden_dims: int | list[int] | None = None,
        activation: Callable[[jnp.ndarray], jnp.ndarray] = nnx.relu,
    ):
        super().__init__(obs_shape=obs_shape, action_dim=action_dim)
        if num_lstm_layers < 1:
            raise ValueError("num_lstm_layers must be >= 1")
        if sequence_pooling not in {"last", "mean", "max"}:
            raise ValueError("sequence_pooling must be one of: 'last', 'mean', 'max'")

        self.hidden_size = int(hidden_size)
        self.num_lstm_layers = int(num_lstm_layers)
        self.bidirectional = bool(bidirectional)
        self.sequence_pooling = sequence_pooling
        self.activation = activation
        self.head_hidden_dims = (
            []
            if head_hidden_dims is None
            else [head_hidden_dims]
            if isinstance(head_hidden_dims, int)
            else list(head_hidden_dims)
        )

        feature_dim = self.input_dim
        policy_dim = self.action_dim

        self.lstm_layers_fwd = nnx.List(
            [
                nnx.OptimizedLSTMCell(
                    in_features=feature_dim if i == 0 else self.hidden_size,
                    hidden_features=self.hidden_size,
                    rngs=rngs,
                )
                for i in range(self.num_lstm_layers)
            ]
        )

        self.lstm_layers_bwd = (
            nnx.List(
                [
                    nnx.OptimizedLSTMCell(
                        in_features=feature_dim if i == 0 else self.hidden_size,
                        hidden_features=self.hidden_size,
                        rngs=rngs,
                    )
                    for i in range(self.num_lstm_layers)
                ]
            )
            if self.bidirectional
            else None
        )

        encoded_dim = self.hidden_size * (2 if self.bidirectional else 1)
        if self.head_hidden_dims:
            self.head_mlp = nnx.Sequential(
                *[
                    item
                    for i, h in enumerate(self.head_hidden_dims)
                    for item in (
                        nnx.Linear(
                            in_features=encoded_dim if i == 0 else self.head_hidden_dims[i - 1],
                            out_features=h,
                            rngs=rngs,
                        ),
                        activation,
                    )
                ]
            )
            head_out_dim = self.head_hidden_dims[-1]
        else:
            self.head_mlp = None
            head_out_dim = encoded_dim

        self.policy_head = nnx.Linear(
            in_features=head_out_dim,
            out_features=policy_dim,
            kernel_init=orthogonal_init(0.01),
            rngs=rngs,
        )
        self.value_head = nnx.Linear(
            in_features=head_out_dim,
            out_features=1,
            kernel_init=orthogonal_init(1.0),
            rngs=rngs,
        )

    def _run_lstm_stack(
        self, x: jnp.ndarray, cells: nnx.List[nnx.OptimizedLSTMCell]
    ) -> jnp.ndarray:
        """Run stacked LSTM over sequence.

        Parameters
        ----------
        x
            Input sequence in shape (B, T, F).
        cells
            LSTM cell stack for one direction.

        Returns
        -------
        jnp.ndarray
            Sequence outputs of shape (B, T, H).
        """
        batch_size, _, _ = x.shape
        x_tb = jnp.transpose(x, (1, 0, 2))  # (T, B, F)

        out = x_tb
        for cell in cells:
            h0 = jnp.zeros((batch_size, self.hidden_size), dtype=x.dtype)
            c0 = jnp.zeros((batch_size, self.hidden_size), dtype=x.dtype)

            def step(carry, xt):
                h, c = carry
                (h_new, c_new), _ = cell((h, c), xt)
                return (h_new, c_new), h_new

            (_, _), out = jax.lax.scan(step, (h0, c0), out)

        return jnp.transpose(out, (1, 0, 2))  # (B, T, H)

    def _pool_sequence(self, seq: jnp.ndarray) -> jnp.ndarray:
        if self.sequence_pooling == "last":
            return seq[:, -1, :]
        if self.sequence_pooling == "mean":
            return jnp.mean(seq, axis=1)
        return jnp.max(seq, axis=1)

    def encode(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Encode observation into a feature vector via LSTM.

        Handles batched observations ``[B, *obs_shape]`` by flattening spatial
        dims and running through LSTM stack + pooling + optional head MLP.

        Parameters
        ----------
        obs
            Batched observation ``[B, *obs_shape]``.

        Returns
        -------
        jnp.ndarray
            Feature vector ``[B, D]``.
        """
        x = obs.reshape(obs.shape[0], 1, -1)  # [B, 1, F]

        out_fwd = self._run_lstm_stack(x, self.lstm_layers_fwd)
        if self.lstm_layers_bwd is not None:
            out_bwd = self._run_lstm_stack(jnp.flip(x, axis=1), self.lstm_layers_bwd)
            out_bwd = jnp.flip(out_bwd, axis=1)
            out_seq = jnp.concatenate([out_fwd, out_bwd], axis=-1)
        else:
            out_seq = out_fwd

        z = self._pool_sequence(out_seq)
        if self.head_mlp is not None:
            z = self.head_mlp(z)

        return z

    def decode(self, core_output: jnp.ndarray) -> ModelOutput:
        """Compute policy logits and value from feature vector."""
        return ModelOutput(
            value=self.value_head(core_output),
            policy_logits=self.policy_head(core_output),
        )

    def __call__(self, obs: jnp.ndarray) -> ModelOutput:
        """Forward pass with exotic input shape handling.

        Supports 1D-5D inputs:
        - 1D ``[F]``: single flat obs (unbatched)
        - 2D ``[B, F]``: batched flat obs
        - 3D ``[H, W, C]``: single spatial obs (unbatched)
        - 4D ``[B, H, W, C]``: batched spatial obs
        - 5D ``[B, T, H, W, C]``: batched sequential spatial obs
        """
        squeeze_output = False

        if obs.ndim == 1:
            x = obs[None, None, :]
            squeeze_output = True
        elif obs.ndim == 2:
            x = obs[:, None, :]
        elif obs.ndim == 3:
            x = obs.reshape(1, 1, -1)
            squeeze_output = True
        elif obs.ndim == 4:
            batch_size = obs.shape[0]
            x = obs.reshape(batch_size, 1, -1)
        elif obs.ndim == 5:
            batch_size, seq_len = obs.shape[0], obs.shape[1]
            x = obs.reshape(batch_size, seq_len, -1)
        else:
            raise ValueError(
                f"Expected obs to have 1-5 dimensions, got {obs.ndim} with shape {obs.shape}"
            )

        out_fwd = self._run_lstm_stack(x, self.lstm_layers_fwd)
        if self.lstm_layers_bwd is not None:
            out_bwd = self._run_lstm_stack(jnp.flip(x, axis=1), self.lstm_layers_bwd)
            out_bwd = jnp.flip(out_bwd, axis=1)
            out_seq = jnp.concatenate([out_fwd, out_bwd], axis=-1)
        else:
            out_seq = out_fwd

        z = self._pool_sequence(out_seq)
        if self.head_mlp is not None:
            z = self.head_mlp(z)

        output = self.decode(z)

        if squeeze_output:
            output = jax.tree.map(lambda x: x.squeeze(0), output)
        return output
