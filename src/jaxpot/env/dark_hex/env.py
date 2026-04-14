"""PGX-compatible Dark Hex environment wrapper.

Imperfect-information Hex: players can only see their own stones
and cells revealed through failed placement attempts.

Two variants:
  - classical (default): failed move reveals cell, same player retries
  - abrupt: failed move reveals cell, turn passes to opponent
"""

from typing import Literal

import jax
import jax.numpy as jnp
import pgx.core as core
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

from .game import Game, GameState
from .observation import OBSERVATION_CLASSES

_DEFAULT_NUM_CELLS = 9

_DEFAULT_GAME_STATE = GameState(
    color=jnp.int32(0),
    board=jnp.zeros(_DEFAULT_NUM_CELLS, dtype=jnp.int32),
    black_view=jnp.zeros(_DEFAULT_NUM_CELLS, dtype=jnp.int32),
    white_view=jnp.zeros(_DEFAULT_NUM_CELLS, dtype=jnp.int32),
    winner=jnp.int32(-1),
    move_succeeded=jnp.bool_(True),
)


@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    observation: Array = jnp.zeros(_DEFAULT_NUM_CELLS * 3, dtype=jnp.float32)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = jnp.bool_(False)
    truncated: Array = jnp.bool_(False)
    legal_action_mask: Array = jnp.ones(_DEFAULT_NUM_CELLS, dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)
    _player_order: Array = jnp.int32([0, 1])
    _x: GameState = _DEFAULT_GAME_STATE

    @property
    def env_id(self) -> core.EnvId:
        return "dark_hex"

    def to_svg(
        self,
        *,
        color_theme: Literal["light", "dark"] | None = None,
        scale: "float | None" = None,
    ) -> str:
        from jaxpot.env.visualizer import Visualizer

        v = Visualizer(color_theme=color_theme, scale=scale)
        return v.get_dwg(states=self).tostring()


class DarkHex(core.Env):
    """PGX-compatible Dark Hex environment.

    Two-player imperfect-information game on a hex grid. Players take
    turns attempting to place their stone. If the cell is already occupied,
    the true contents are revealed but (in classical mode) the player
    must try again.

    Player 0 (Black) wins by connecting North edge to South edge.
    Player 1 (White) wins by connecting West edge to East edge.

    Action space: num_rows * num_cols discrete actions (cell indices, row-major).

    Observation: (num_rows, num_cols, 3) float32 array (perspective-normalized)
      Channel 0: Empty cells in player's view
      Channel 1: Player's own stones
      Channel 2: Opponent stones (revealed through failed attempts)

    Args:
        num_rows: Board height. Default 3.
        num_cols: Board width. Default 3.
        observation_cls: Observation type ("default" for spatial, "flat" for 1-d).
        abrupt: If True, use abrupt variant (failed move loses turn).
        max_steps: Maximum step count before truncation.
    """

    def __init__(
        self,
        num_rows: int = 3,
        num_cols: int = 3,
        observation_cls: str = "default",
        abrupt: bool = False,
        max_steps: int = 50,
    ):
        super().__init__()
        self._game = Game(num_rows=num_rows, num_cols=num_cols, abrupt=abrupt)
        self._num_rows = num_rows
        self._num_cols = num_cols
        self._max_steps = max_steps
        self._observation_cls = OBSERVATION_CLASSES[observation_cls]

    def _init(self, key: PRNGKey) -> State:
        _player_order = jnp.array([[0, 1], [1, 0]])[jax.random.bernoulli(key).astype(jnp.int32)]
        x = self._game.init()
        num_cells = self._game.num_cells
        legal_action_mask = self._game.legal_action_mask(x)
        return State(
            current_player=_player_order[x.color],
            observation=jnp.zeros(num_cells * 3, dtype=jnp.float32),
            legal_action_mask=legal_action_mask,
            _player_order=_player_order,
            _x=x,
        )

    def _step(self, state, action: Array, key) -> "core.State":
        x = self._game.step(state._x, action)

        terminated = self._game.is_terminal(x)
        rewards = self._game.rewards(x)[state._player_order]
        rewards = jax.lax.select(terminated, rewards, jnp.zeros(2, jnp.float32))

        legal_action_mask = self._game.legal_action_mask(x)

        # Truncation
        should_truncate = (state._step_count >= self._max_steps) & ~terminated
        rewards = jnp.where(should_truncate, jnp.float32([0.0, 0.0]), rewards)

        return state.replace(
            current_player=state._player_order[x.color],
            legal_action_mask=legal_action_mask,
            rewards=rewards,
            terminated=terminated | should_truncate,
            truncated=should_truncate,
            _x=x,
        )

    def _observe(self, state, player_id: Array) -> Array:
        my_color = jax.lax.select(state._player_order[0] == player_id, jnp.int32(0), jnp.int32(1))
        return self._observation_cls.from_state(
            state._x,
            color=my_color,
            num_rows=self._num_rows,
            num_cols=self._num_cols,
        )

    @property
    def id(self) -> core.EnvId:
        return "dark_hex"

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 2
