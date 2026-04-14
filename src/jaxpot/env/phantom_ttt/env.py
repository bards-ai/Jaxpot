"""PGX-compatible Phantom Tic-Tac-Toe environment wrapper.

Imperfect information tic-tac-toe: players can only see their own marks
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

from .game import NUM_ACTIONS, Game, GameState
from .observation import OBSERVATION_CLASSES


@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    observation: Array = jnp.zeros((3, 3, 3), dtype=jnp.float32)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = jnp.bool_(False)
    truncated: Array = jnp.bool_(False)
    legal_action_mask: Array = jnp.ones(NUM_ACTIONS, dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)
    _player_order: Array = jnp.int32([0, 1])
    _x: GameState = GameState()

    @property
    def env_id(self) -> core.EnvId:
        return "phantom_ttt"

    def to_svg(
        self,
        *,
        color_theme: Literal["light", "dark"] | None = None,
        scale: "float | None" = None,
    ) -> str:
        from jaxpot.env.visualizer import Visualizer

        v = Visualizer(color_theme=color_theme, scale=scale)
        return v.get_dwg(states=self).tostring()


class PhantomTTT(core.Env):
    """PGX-compatible Phantom Tic-Tac-Toe environment.

    Two-player imperfect information game on a 3x3 grid. Players take
    turns attempting to place their mark. If the cell is already occupied,
    the true contents are revealed but (in classical mode) the player
    must try again.

    Action space: 9 discrete actions (cells 0-8, row-major)
      0-2: top row (left to right)
      3-5: middle row
      6-8: bottom row

    Observation: (3, 3, 3) float32 array (perspective-normalized)
      Channel 0: Empty cells in player's view
      Channel 1: Player's own marks
      Channel 2: Opponent marks (revealed through failed attempts)

    Args:
        observation_cls: Observation type ("default" for spatial, "flat" for 27-d).
        abrupt: If True, use abrupt variant (failed move loses turn).
        max_steps: Maximum step count before truncation.
    """

    def __init__(
        self,
        observation_cls: str = "default",
        abrupt: bool = False,
        max_steps: int = 50,
    ):
        super().__init__()
        self._game = Game(abrupt=abrupt)
        self._observation_cls = OBSERVATION_CLASSES[observation_cls]
        self._max_steps = max_steps

    def _init(self, key: PRNGKey) -> State:
        _player_order = jnp.array([[0, 1], [1, 0]])[jax.random.bernoulli(key).astype(jnp.int32)]
        x = self._game.init()
        legal_action_mask = self._game.legal_action_mask(x)
        return State(
            current_player=_player_order[x.color],
            _player_order=_player_order,
            _x=x,
            legal_action_mask=legal_action_mask,
        )

    def _step(self, state: State, action: Array, key) -> State:
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

    def _observe(self, state: State, player_id: Array) -> Array:
        my_color = jax.lax.select(state._player_order[0] == player_id, jnp.int32(0), jnp.int32(1))
        return self._observation_cls.from_state(state._x, color=my_color)

    @property
    def id(self) -> core.EnvId:
        return "phantom_ttt"

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 2
