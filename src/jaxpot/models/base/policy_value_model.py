from __future__ import annotations

import math
from abc import abstractmethod
from typing import Any

import jax.numpy as jnp
from flax import nnx

from jaxpot.models.base.outputs import ModelOutput
from jaxpot.models.utils.shapes import normalize_action_dim, normalize_obs_shape


class PolicyValueModel(nnx.Module):
    """Minimal runtime contract for policy/value models."""

    def __init__(
        self,
        obs_shape: int | tuple[int, ...] = (),
        action_dim: int | tuple[int, ...] = 0,
        **kwargs: Any,
    ):
        del kwargs
        self.obs_shape = normalize_obs_shape(obs_shape)
        self.action_dim = normalize_action_dim(action_dim)

    @property
    def input_dim(self) -> int:
        """Flattened observation size when the observation is array-like."""
        return math.prod(self.obs_shape)

    @property
    def hidden_shape(self) -> tuple[int, ...]:
        """Per-sample shape of the recurrent state (batch dim excluded).

        Non-recurrent models return ``(1,)`` so the rollout buffer can still
        allocate a dummy slot. Recurrent subclasses should override this with
        the model-defined opaque state shape (e.g. ``(2, num_layers, hidden)``
        for LSTMs that pack ``(h, c)`` on axis 0).
        """
        return (1,)

    def init_state(self, batch_size: int) -> jnp.ndarray:
        """Initialize the recurrent state as a single batch-first tensor.

        Returns ``[batch_size, *hidden_shape]``. Override in recurrent
        subclasses if a non-zero initial state is needed.
        """
        return jnp.zeros((batch_size, *self.hidden_shape))

    @property
    def is_recurrent(self) -> bool:
        """Whether the model consumes and emits recurrent state."""
        return False

    @abstractmethod
    def __call__(
        self,
        obs: Any,
        hidden_state: jnp.ndarray | None = None,
    ) -> ModelOutput:
        """Run a forward pass and return policy/value outputs."""
        raise NotImplementedError
