from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from jaxpot.rollout.buffer import RolloutBuffer


class AuxTargetHook(ABC):
    """Collects one auxiliary target field from rollout state."""

    target_field: str

    @abstractmethod
    def init_buffer(self, num_envs: int, num_steps: int) -> jnp.ndarray:
        """Return a [num_envs, num_steps, 2, ...] sentinel-initialized target buffer."""
        raise NotImplementedError

    @abstractmethod
    def collect(self, state, current_player: jnp.ndarray, opp_player: jnp.ndarray) -> jnp.ndarray:
        """Return per-env target values for the acting player at the current step."""
        raise NotImplementedError

    @abstractmethod
    def update_buffer(
        self,
        rollout_buffer: "RolloutBuffer",
        batch_indexing: jnp.ndarray,
        time_player_idx: jnp.ndarray,
        current_player: jnp.ndarray,
        target_values: jnp.ndarray,
    ) -> "RolloutBuffer":
        """Write collected targets into the rollout buffer and return updated buffer."""
        raise NotImplementedError


class GameProgressTargetHook(AuxTargetHook):
    """Collects game progress (step_count / max_steps) as auxiliary target."""

    target_field = "game_progress_target"

    def __init__(self, max_steps: int = 42):
        self.max_steps = max_steps

    def init_buffer(self, num_envs: int, num_steps: int) -> jnp.ndarray:
        return jnp.full((num_envs, num_steps, 2, 1), -1.0, dtype=jnp.float32)

    def collect(self, state, current_player: jnp.ndarray, opp_player: jnp.ndarray) -> jnp.ndarray:
        progress = state._step_count / self.max_steps
        # Broadcast to per-env scalar
        return jnp.broadcast_to(progress, current_player.shape).astype(jnp.float32)

    def update_buffer(
        self,
        rollout_buffer: "RolloutBuffer",
        batch_indexing: jnp.ndarray,
        time_player_idx: jnp.ndarray,
        current_player: jnp.ndarray,
        target_values: jnp.ndarray,
    ) -> "RolloutBuffer":
        current_targets = rollout_buffer.get_aux_target(self.target_field)
        updated_targets = current_targets.at[
            batch_indexing, time_player_idx, current_player, 0
        ].set(target_values)
        return rollout_buffer.set_aux_target(self.target_field, updated_targets)
