from __future__ import annotations

import jax.numpy as jnp
from flax import struct


class ModelOutput(struct.PyTreeNode):
    """Base model output for policy/value inference."""

    value: jnp.ndarray
    policy_logits: jnp.ndarray
    hidden_state: jnp.ndarray | None = None


@struct.dataclass
class GameProgressModelOutput(ModelOutput):
    game_progress: jnp.ndarray | None = None
