"""PGX-compatible Quoridor environment wrapper.

Perspective normalization: color 1's view is rotated 180° so both colors
see themselves at row 0 with the opponent at row 8. This makes the action
space symmetric — "North" (action 1) always means "toward opponent's base".
"""

from typing import Literal

import jax
import jax.numpy as jnp
import pgx.core as core
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

from .game import BOARD_SIZE, DEFAULT_MAX_STEPS, NUM_ACTIONS, Game, GameState
from .observation import OBSERVATION_CLASSES

# 180° rotation remapping for movement actions (12 total).
# Cardinal: S(0)<->N(1), E(2)<->W(3)
# Jump:     S(4)<->N(5), E(6)<->W(7)
# Diagonal: SE(8)<->NW(11), SW(9)<->NE(10)
_MOVE_FLIP = jnp.array([1, 0, 3, 2, 5, 4, 7, 6, 11, 10, 9, 8], dtype=jnp.int32)


def _flip_action(action: Array) -> Array:
    """Transform canonical action to absolute coords (180° rotation)."""
    is_move = action < 12
    flipped_move = _MOVE_FLIP[jnp.clip(action, 0, 11)]

    # Walls: index i in 8x8 grid -> 63-i  (flips both row and col)
    is_h_wall = (action >= 12) & (action < 76)
    h_idx = jnp.clip(action - 12, 0, 63)
    v_idx = jnp.clip(action - 76, 0, 63)
    wall_action = jnp.where(is_h_wall, 75 - h_idx, 139 - v_idx)

    return jnp.where(is_move, flipped_move, wall_action)


def _flip_mask(mask: Array) -> Array:
    """Transform legal action mask from absolute to canonical coords."""
    return jnp.concatenate(
        [
            mask[:12][_MOVE_FLIP],
            jnp.flip(mask[12:76]),
            jnp.flip(mask[76:]),
        ]
    )


@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    observation: Array = jnp.zeros((BOARD_SIZE, BOARD_SIZE, 4), dtype=jnp.float32)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = jnp.bool_(False)
    truncated: Array = jnp.bool_(False)
    legal_action_mask: Array = jnp.ones(NUM_ACTIONS, dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)
    _player_order: Array = jnp.int32([0, 1])
    _x: GameState = GameState()

    @property
    def env_id(self) -> core.EnvId:
        return "quoridor"

    def to_svg(
        self,
        *,
        color_theme: Literal["light", "dark"] | None = None,
        scale: float | None = None,
    ) -> str:
        from jaxpot.env.visualizer import Visualizer

        v = Visualizer(color_theme=color_theme, scale=scale)
        return v.get_dwg(states=self).tostring()


class Quoridor(core.Env):
    """PGX-compatible Quoridor environment.

    Two-player board game on a 9x9 grid. Players race to reach the
    opposite side while placing walls to impede the opponent.

    The action space is perspective-normalized: both colors see the board
    from the same orientation (my pawn near row 0, opponent near row 8).
    "North" (action 1) always means "toward opponent's base" for both colors.

    Action space: 140 discrete actions
      0-3:    Cardinal moves (S, N, E, W)
      4-7:    Straight jumps over opponent
      8-11:   Diagonal jumps (when straight jump blocked)
      12-75:  Horizontal wall placements (8x8 grid)
      76-139: Vertical wall placements (8x8 grid)

    Observation: (9, 9, 6) float32 array
      Channel 0: My pawn position
      Channel 1: Opponent pawn position
      Channel 2: Horizontal walls
      Channel 3: Vertical walls
      Channel 4: My walls remaining (normalized)
      Channel 5: Opponent walls remaining (normalized)
    """

    def __init__(
        self,
        observation_cls: str = "default",
        speed_reward_scale: float = 0.0,
        max_steps: int = DEFAULT_MAX_STEPS,
    ):
        super().__init__()
        self._game = Game()
        self._observation_cls = OBSERVATION_CLASSES[observation_cls]
        self._speed_reward_scale = speed_reward_scale
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

    def _step(self, state: core.State, action: Array, key) -> State:
        assert isinstance(state, State)

        # Un-flip action for color 1 (canonical -> absolute)
        needs_flip = state._x.color == 1
        abs_action = jax.lax.select(needs_flip, _flip_action(action), action)

        x = self._game.step(state._x, abs_action)

        legal_action_mask = self._game.legal_action_mask(x)
        # Flip legal mask for color 1 (absolute -> canonical)
        next_needs_flip = x.color == 1
        legal_action_mask = jax.lax.select(
            next_needs_flip, _flip_mask(legal_action_mask), legal_action_mask
        )

        terminated = self._game.is_terminal(x)
        rewards = self._game.rewards(x)[state._player_order]
        rewards = jax.lax.select(terminated, rewards, jnp.zeros(2, jnp.float32))

        # Speed bonus: reward winning faster (only applied to positive reward)
        if self._speed_reward_scale > 0:
            speed_bonus = self._speed_reward_scale * (1.0 - state._step_count / self._max_steps)
            speed_bonus = jax.lax.select(terminated, speed_bonus, jnp.float32(0.0))
            rewards = rewards + jnp.maximum(rewards, 0.0) * speed_bonus

        # Truncation: if max steps reached without a winner, it's a draw
        should_truncate = (state._step_count >= self._max_steps) & ~terminated
        trunc_rewards = jnp.float32([0.0, 0.0])

        rewards = jnp.where(should_truncate, trunc_rewards, rewards)
        truncated = should_truncate

        return state.replace(
            current_player=state._player_order[x.color],
            legal_action_mask=legal_action_mask,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            _x=x,
        )

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        my_color = jax.lax.select(state._player_order[0] == player_id, jnp.int32(0), jnp.int32(1))
        flip = my_color == 1
        return self._observation_cls.from_state(
            state._x, color=my_color, step_count=state._step_count, flip=flip
        )

    @property
    def id(self) -> core.EnvId:
        return "quoridor"

    @property
    def version(self) -> str:
        return "v1"

    @property
    def num_players(self) -> int:
        return 2
