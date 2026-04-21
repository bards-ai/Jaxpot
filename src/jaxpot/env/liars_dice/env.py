"""PGX-compatible Liar's Dice environment wrapper.

Two-player imperfect information dice game. Each player privately rolls
dice, then players alternate making increasing bids on total dice counts
or calling "Liar!" to challenge.

Default configuration: 2 players, 5 dice each, 6-sided dice.
Action space: total_dice * 6 + 1 (all possible bids + liar call).
"""

from typing import Literal

import jax
import jax.numpy as jnp
import pgx.core as core
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

from .game import Game, GameState
from .observation import OBSERVATION_CLASSES


@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    observation: Array = jnp.zeros(90, dtype=jnp.float32)  # default for 5 dice
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = jnp.bool_(False)
    truncated: Array = jnp.bool_(False)
    legal_action_mask: Array = jnp.ones(61, dtype=jnp.bool_)  # default for 5 dice
    _step_count: Array = jnp.int32(0)
    _player_order: Array = jnp.int32([0, 1])
    _x: GameState = GameState(dice=jnp.zeros((2, 5), dtype=jnp.int32))

    @property
    def env_id(self) -> core.EnvId:
        return "liars_dice"

    def to_svg(
        self,
        *,
        color_theme: Literal["light", "dark"] | None = None,
        scale: float | None = None,
    ) -> str:
        from jaxpot.env.visualizer import Visualizer

        v = Visualizer(color_theme=color_theme, scale=scale)
        return v.get_dwg(states=self).tostring()


class LiarsDice(core.Env):
    """PGX-compatible Liar's Dice environment.

    Two-player imperfect information game where each player rolls dice
    privately and then makes increasing bids on the total count of a
    face value across all dice. The highest face (6) is wild.

    A player can either raise the bid or call "Liar!" to challenge.
    When challenged, dice are revealed and the bid is verified.

    Action space: total_dice * 6 + 1 discrete action IDs.
      0 .. total_dice*6-1: bid actions, where action a means
      quantity = a // 6 + 1 and face = a % 6 + 1
      For example: 0 -> (1, 1), 1 -> (1, 2), 5 -> (1, 6), 6 -> (2, 1)
      total_dice*6: call "Liar!" to challenge the previous bid

    Args:
        num_dice: Number of dice per player (default: 5).
        observation_cls: Observation type ("default" or "compact").
        max_steps: Maximum steps before truncation.
    """

    def __init__(
        self,
        num_dice: int = 5,
        observation_cls: str = "default",
        max_steps: int = 200,
    ):
        super().__init__()
        self._game = Game(num_dice=num_dice)
        self._num_dice = num_dice
        obs_cls = OBSERVATION_CLASSES[observation_cls]
        self._observation_cls = obs_cls(num_dice=num_dice)
        self._max_steps = max_steps

    def _init(self, key: PRNGKey) -> State:
        k1, k2 = jax.random.split(key)
        _player_order = jnp.array([[0, 1], [1, 0]])[jax.random.bernoulli(k1).astype(jnp.int32)]
        x = self._game.init(k2)
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
        # _player_order maps internal color -> external player.
        # We need the inverse: external player -> internal color.
        my_color = jax.lax.select(state._player_order[0] == player_id, jnp.int32(0), jnp.int32(1))
        return self._observation_cls.from_state(state._x, color=my_color, num_dice=self._num_dice)

    @property
    def id(self) -> core.EnvId:
        return "liars_dice"

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 2
