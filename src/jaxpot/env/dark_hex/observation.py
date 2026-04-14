"""Observation classes for Dark Hex."""

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from jaxpot.env.observation import ArrayObservation, Observation


class DarkHexObservation(ArrayObservation):
    """Spatial observation: (num_rows, num_cols, 3) float32 array.

    The observation is from the current player's perspective.
    Both players see themselves as "mine" regardless of whether they're Black or White.

    Channels:
      0: Empty cells (in this player's view)
      1: My stones (cells I've placed)
      2: Opponent stones (cells revealed by failed moves)
    """

    # Default shape for 3x3 board; actual shape depends on board size
    @property
    def shape(self) -> tuple[int, ...]:
        return (3, 3, 3)

    @classmethod
    def from_state(
        cls,
        state,
        *,
        color: Array,
        num_rows: int = 3,
        num_cols: int = 3,
        **_kwargs,
    ) -> Array:
        view = jax.lax.select(color == 0, state.black_view, state.white_view)
        my_mark = color + 1  # 1 for Black, 2 for White
        opp_mark = 2 - color  # 2 for Black's opponent, 1 for White's opponent

        empty = (view == 0).astype(jnp.float32).reshape(num_rows, num_cols)
        mine = (view == my_mark).astype(jnp.float32).reshape(num_rows, num_cols)
        theirs = (view == opp_mark).astype(jnp.float32).reshape(num_rows, num_cols)

        return jnp.stack([empty, mine, theirs], axis=-1)


class DarkHexFlatObservation(ArrayObservation):
    """Flat one-hot observation: (num_cells * 3,) float32 array.

    For each cell, 3 bits: [empty, mine, opponent].
    Same perspective normalization as the spatial version.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        return (27,)

    @classmethod
    def from_state(
        cls,
        state,
        *,
        color: Array,
        num_rows: int = 3,
        num_cols: int = 3,
        **_kwargs,
    ) -> Array:
        view = jax.lax.select(color == 0, state.black_view, state.white_view)
        my_mark = color + 1
        opp_mark = 2 - color

        empty = (view == 0).astype(jnp.float32)
        mine = (view == my_mark).astype(jnp.float32)
        theirs = (view == opp_mark).astype(jnp.float32)

        # Interleave: [cell0_empty, cell0_mine, cell0_opp, cell1_empty, ...]
        return jnp.stack([empty, mine, theirs], axis=-1).reshape(-1)


OBSERVATION_CLASSES: dict[str, type[Observation[Any]]] = {
    "default": DarkHexObservation,
    "flat": DarkHexFlatObservation,
}
