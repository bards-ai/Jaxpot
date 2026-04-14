"""Observation classes for Quoridor."""

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from jaxpot.env.observation import ArrayObservation, Observation

from .game import BOARD_SIZE, WALL_SIZE


NUM_SPATIAL_CHANNELS = 4
NUM_SCALAR_FEATURES = 2
SPATIAL_SIZE = BOARD_SIZE * BOARD_SIZE * NUM_SPATIAL_CHANNELS  # 324


def _build_spatial_planes(state, *, color: Array, flip: Array) -> Array:
    """Build the 4-channel spatial observation (9, 9, 4)."""
    opp_color = 1 - color
    my_pos = state.pawn_pos[color]
    opp_pos = state.pawn_pos[opp_color]

    # 180° rotation for color 1: flat pos (r*9+c) -> (8-r)*9+(8-c) = 80-pos
    my_pos = jax.lax.select(flip, 80 - my_pos, my_pos)
    opp_pos = jax.lax.select(flip, 80 - opp_pos, opp_pos)

    my_plane = (
        jnp.zeros(BOARD_SIZE * BOARD_SIZE, dtype=jnp.float32)
        .at[my_pos]
        .set(1.0)
        .reshape(BOARD_SIZE, BOARD_SIZE)
    )
    opp_plane = (
        jnp.zeros(BOARD_SIZE * BOARD_SIZE, dtype=jnp.float32)
        .at[opp_pos]
        .set(1.0)
        .reshape(BOARD_SIZE, BOARD_SIZE)
    )

    # Flip walls in 8x8 grid before padding to 9x9
    h_walls = jax.lax.select(flip, jnp.flip(state.h_walls), state.h_walls)
    v_walls = jax.lax.select(flip, jnp.flip(state.v_walls), state.v_walls)

    h_plane = (
        jnp.zeros((BOARD_SIZE, BOARD_SIZE), dtype=jnp.float32)
        .at[:WALL_SIZE, :WALL_SIZE]
        .set(h_walls.astype(jnp.float32))
    )
    v_plane = (
        jnp.zeros((BOARD_SIZE, BOARD_SIZE), dtype=jnp.float32)
        .at[:WALL_SIZE, :WALL_SIZE]
        .set(v_walls.astype(jnp.float32))
    )

    return jnp.stack([my_plane, opp_plane, h_plane, v_plane], axis=-1)


class QuoridorSpatialObservation(ArrayObservation):
    """Spatial observation: (9, 9, 4) float32 array.

    Both colors see the board from the same perspective via 180° rotation
    for color 1: my pawn near row 0, opponent near row 8, forward = S.

    Channels:
      0: My pawn position (one-hot)
      1: Opponent pawn position (one-hot)
      2: Horizontal walls (8x8 zero-padded to 9x9)
      3: Vertical walls (8x8 zero-padded to 9x9)
    """

    @property
    def shape(self) -> tuple[int, ...]:
        return (BOARD_SIZE, BOARD_SIZE, NUM_SPATIAL_CHANNELS)

    @classmethod
    def from_state(cls, state, *, color: Array, flip: Array = jnp.bool_(False), **_kwargs) -> Array:
        return _build_spatial_planes(state, color=color, flip=flip)


class QuoridorSpatialScalarObservation(ArrayObservation):
    """Spatial observation with scalar features, flattened into a single vector.

    Layout: [spatial_flat (324,), scalar (2,)] = (326,) total.

    Spatial channels (9, 9, 4) — same as QuoridorSpatialObservation.
    Scalar features (2,):
      0: My walls remaining / 10
      1: Opponent walls remaining / 10
    """

    @property
    def shape(self) -> tuple[int, ...]:
        return (SPATIAL_SIZE + NUM_SCALAR_FEATURES,)

    @classmethod
    def from_state(cls, state, *, color: Array, flip: Array = jnp.bool_(False), **_kwargs) -> Array:
        opp_color = 1 - color
        spatial = _build_spatial_planes(state, color=color, flip=flip)

        my_walls = state.walls_remaining[color] / 10.0
        opp_walls = state.walls_remaining[opp_color] / 10.0

        return jnp.concatenate([spatial.reshape(-1), jnp.array([my_walls, opp_walls])])


OBSERVATION_CLASSES: dict[str, type[Observation[Any]]] = {
    "default": QuoridorSpatialObservation,
    "spatial_scalar": QuoridorSpatialScalarObservation,
}
