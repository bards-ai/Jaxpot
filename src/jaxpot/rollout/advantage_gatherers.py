from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp

from jaxpot.rl.utils import (
    per_seat_returns_from_rollout_buffer,
    gae_advantages_from_rollout_buffer,
)
from jaxpot.rollout.buffer import RolloutBuffer


class AdvantageGatherer(Protocol):
    """Strategy for constructing advantages and optional value-loss masks."""

    def gather(
        self, rollout_buffer: RolloutBuffer, seats: tuple[int, ...], num_steps: int
    ) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...] | None]:
        ...


@dataclass(frozen=True)
class GAEAdvantageGatherer:
    gamma: float
    gae_lambda: float

    def gather(
        self, rollout_buffer: RolloutBuffer, seats: tuple[int, ...], num_steps: int
    ) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...] | None]:
        del num_steps
        advs = tuple(
            gae_advantages_from_rollout_buffer(
                rollout_buffer, seat=seat, gamma=self.gamma, gae_lambda=self.gae_lambda
            )
            for seat in seats
        )
        return advs, None


@dataclass(frozen=True)
class PerSeatReturnGatherer:
    """Per-seat terminal return gatherer equivalent to PGX AlphaZero value targets.

    Computes value targets via a per-seat backward scan with positive ``backup_gamma``
    as the discount.  For 2-player zero-sum alternating-move games this gives identical
    targets to the PGX reference's global-timeline alternating backup (discount=-1),
    because each seat only processes its own steps and its own signed terminal reward
    (opponent rewards are back-filled by the rollout collector).

    Use ``backup_gamma=1.0`` for standard undiscounted games such as Go or Connect4.
    """

    backup_gamma: float

    def gather(
        self, rollout_buffer: RolloutBuffer, seats: tuple[int, ...], num_steps: int
    ) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...] | None]:
        advs: list[jax.Array] = []
        masks: list[jax.Array] = []
        for seat in seats:
            ret, msk = per_seat_returns_from_rollout_buffer(
                rollout_buffer, seat, num_steps, self.backup_gamma
            )
            advs.append(ret - rollout_buffer.value[:, :, seat])
            masks.append(msk.astype(jnp.float32))
        return tuple(advs), tuple(masks)

