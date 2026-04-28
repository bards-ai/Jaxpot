"""Observation classes for Phantom Tic-Tac-Toe."""

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from jaxpot.env.observation import ArrayObservation, Observation


class PhantomTTTObservation(ArrayObservation):
    """Spatial observation: (3, 3, 3) float32 array.

    The observation is from the current player's perspective.
    Both players see themselves as "mine" regardless of whether they're X or O.

    Channels:
      0: Empty cells (in this player's view)
      1: My marks (cells I've placed)
      2: Opponent marks (cells revealed by failed moves)
    """

    @property
    def shape(self) -> tuple[int, ...]:
        return (3, 3, 3)

    @classmethod
    def from_state(cls, state, *, color: Array, **_kwargs) -> Array:
        # Get this player's view
        view = jax.lax.select(color == 0, state.x_view, state.o_view)
        my_mark = color + 1  # 1 for X, 2 for O
        opp_mark = 2 - color  # 2 for X's opponent, 1 for O's opponent

        empty = (view == 0).astype(jnp.float32).reshape(3, 3)
        mine = (view == my_mark).astype(jnp.float32).reshape(3, 3)
        theirs = (view == opp_mark).astype(jnp.float32).reshape(3, 3)

        return jnp.stack([empty, mine, theirs], axis=-1)


class PhantomTTTFlatObservation(ArrayObservation):
    """Flat one-hot observation: (27,) float32 array.

    For each of 9 cells, 3 bits: [empty, mine, opponent].
    Same perspective normalization as the spatial version.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        return (27,)

    @classmethod
    def from_state(cls, state, *, color: Array, **_kwargs) -> Array:
        view = jax.lax.select(color == 0, state.x_view, state.o_view)
        my_mark = color + 1
        opp_mark = 2 - color

        empty = (view == 0).astype(jnp.float32)
        mine = (view == my_mark).astype(jnp.float32)
        theirs = (view == opp_mark).astype(jnp.float32)

        # Interleave: [cell0_empty, cell0_mine, cell0_opp, cell1_empty, ...]
        return jnp.stack([empty, mine, theirs], axis=-1).reshape(-1)


OBSERVATION_CLASSES: dict[str, type[Observation[Any]]] = {
    "default": PhantomTTTObservation,
    "flat": PhantomTTTFlatObservation,
}
