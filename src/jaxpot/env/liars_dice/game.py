"""Pure JAX game logic for Liar's Dice.

Two players each roll `num_dice` six-sided dice privately.  Players alternate
making bids of the form (quantity, face_value) claiming that at least `quantity`
dice across *both* players show `face_value`. The highest face (6) is **wild**
and counts as matching any bid face.

Bids must strictly increase (by action ID).  A player may instead call "Liar!"
to challenge the previous bid.  All dice are then revealed:
  - If actual count >= bid quantity → bidder wins, challenger loses.
  - If actual count <  bid quantity → challenger wins, bidder loses.

Action encoding (reset-face ordering):
  action_id = (quantity - 1) * DICE_SIDES + (face - 1)
  quantity   = action_id // DICE_SIDES + 1
  face       = action_id %  DICE_SIDES + 1

  Liar action = total_num_dice * DICE_SIDES  (last action)

Total action space size = total_num_dice * DICE_SIDES + 1
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

DICE_SIDES = 6
WILD_FACE = DICE_SIDES  # face value 6 is wild


class GameState(NamedTuple):
    """Internal game state for Liar's Dice."""
    color: Array = jnp.int32(0)  # whose turn (0 or 1)
    # Dice for each player: (2, num_dice) with values 1..DICE_SIDES
    dice: Array = jnp.zeros((2, 1), dtype=jnp.int32)
    # Current highest bid action ID, -1 means no bid yet
    current_bid: Array = jnp.int32(-1)
    # Who made the last bid (-1 = nobody)
    bidder: Array = jnp.int32(-1)
    # -1=ongoing, 0=player0 wins, 1=player1 wins
    winner: Array = jnp.int32(-1)


def _num_actions(num_dice_per_player: int) -> int:
    """Total action space: all possible bids + 1 liar call."""
    total_dice = 2 * num_dice_per_player
    return total_dice * DICE_SIDES + 1


def _liar_action(num_dice_per_player: int) -> int:
    """Action ID for the 'Liar!' call."""
    total_dice = 2 * num_dice_per_player
    return total_dice * DICE_SIDES


def _bid_quantity(action_id: Array) -> Array:
    """Extract bid quantity from action ID."""
    return action_id // DICE_SIDES + 1


def _bid_face(action_id: Array) -> Array:
    """Extract bid face value (1-6) from action ID."""
    return action_id % DICE_SIDES + 1


def _count_matching(dice: Array, face: Array) -> Array:
    """Count dice matching `face` or the wild face across both players.

    Args:
        dice: (2, num_dice) array of dice values 1..DICE_SIDES.
        face: scalar, the bid face value.

    Returns:
        Scalar count of matching dice.
    """
    all_dice = dice.reshape(-1)
    matches = (all_dice == face) | (all_dice == WILD_FACE)
    # If bidding on the wild face itself, don't double-count
    matches = jnp.where(face == WILD_FACE, all_dice == WILD_FACE, matches)
    return jnp.sum(matches)


def _resolve(dice: Array, bid_action: Array) -> Array:
    """Resolve a Liar call. Returns True if the bid was correct (bidder wins)."""
    quantity = _bid_quantity(bid_action)
    face = _bid_face(bid_action)
    actual = _count_matching(dice, face)
    return actual >= quantity


def _legal_action_mask(state: GameState, num_dice_per_player: int) -> Array:
    """Compute legal action mask.

    - All bids with action_id > current_bid are legal.
    - Liar is legal if at least one bid has been made.
    """
    n_actions = _num_actions(num_dice_per_player)
    liar_id = _liar_action(num_dice_per_player)
    action_ids = jnp.arange(n_actions)

    # Bids strictly higher than current bid
    bid_mask = (action_ids < liar_id) & (action_ids > state.current_bid)
    # Liar is legal only if someone has bid
    liar_mask = (action_ids == liar_id) & (state.current_bid >= 0)

    return bid_mask | liar_mask


def _step(state: GameState, action: Array, num_dice_per_player: int) -> GameState:
    """Execute one action.

    Args:
        state: Current game state.
        action: Action ID (bid or liar).
        num_dice_per_player: Number of dice per player.

    Returns:
        New game state.
    """
    liar_id = _liar_action(num_dice_per_player)
    is_liar = action == liar_id

    # --- Liar call: resolve ---
    bid_correct = _resolve(state.dice, state.current_bid)
    # If bid was correct, bidder wins → challenger (current player) loses
    # If bid was wrong, challenger wins
    liar_winner = jnp.where(bid_correct, state.bidder, state.color)

    # --- Normal bid ---
    new_bid = jnp.where(is_liar, state.current_bid, action)
    new_bidder = jnp.where(is_liar, state.bidder, state.color)
    new_winner = jnp.where(is_liar, liar_winner, jnp.int32(-1))
    new_color = 1 - state.color  # always alternate

    return GameState(
        color=new_color,
        dice=state.dice,
        current_bid=new_bid,
        bidder=new_bidder,
        winner=new_winner,
    )


class Game:
    """Pure JAX Liar's Dice game logic."""

    def __init__(self, num_dice: int = 5):
        self._num_dice = num_dice

    @property
    def num_dice(self) -> int:
        return self._num_dice

    @property
    def num_actions(self) -> int:
        return _num_actions(self._num_dice)

    @property
    def liar_action(self) -> int:
        return _liar_action(self._num_dice)

    def init(self, key: Array) -> GameState:
        """Initialize game with random dice rolls."""
        dice = jax.random.randint(
            key, shape=(2, self._num_dice), minval=1, maxval=DICE_SIDES + 1
        )
        return GameState(
            color=jnp.int32(0),
            dice=dice,
            current_bid=jnp.int32(-1),
            bidder=jnp.int32(-1),
            winner=jnp.int32(-1),
        )

    def step(self, state: GameState, action: Array) -> GameState:
        return _step(state, action, self._num_dice)

    def legal_action_mask(self, state: GameState) -> Array:
        return _legal_action_mask(state, self._num_dice)

    def is_terminal(self, state: GameState) -> Array:
        return state.winner >= 0

    def rewards(self, state: GameState) -> Array:
        """Returns (2,) rewards. Winner gets +1, loser gets -1."""
        return jax.lax.switch(
            jnp.clip(state.winner, 0, 1),
            [
                lambda: jnp.float32([1.0, -1.0]),   # player 0 wins
                lambda: jnp.float32([-1.0, 1.0]),   # player 1 wins
            ],
        )
