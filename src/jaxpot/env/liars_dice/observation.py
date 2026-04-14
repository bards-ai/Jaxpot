"""Observation classes for Liar's Dice."""

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from jaxpot.env.observation import ArrayObservation, Observation

from .game import DICE_SIDES


class LiarsDiceObservation(ArrayObservation):
    """Flat observation for Liar's Dice.

    Layout (all float32):
      [0 .. num_dice * DICE_SIDES - 1]:  One-hot encoding of own dice
      [num_dice * DICE_SIDES .. num_dice * DICE_SIDES + num_bids - 1]:
          Binary flags for which bids have been made (bid history)

    Total size = num_dice * DICE_SIDES + total_dice * DICE_SIDES

    Parameters passed via kwargs:
      color: which player's perspective
      num_dice: dice per player
    """

    def __init__(self, num_dice: int = 5):
        self._num_dice = num_dice

    @property
    def shape(self) -> tuple[int, ...]:
        dice_section = self._num_dice * DICE_SIDES
        total_dice = 2 * self._num_dice
        bid_section = total_dice * DICE_SIDES
        return (dice_section + bid_section,)

    @classmethod
    def from_state(cls, state, *, color: Array, num_dice: int, **_kwargs) -> Array:
        total_dice = 2 * num_dice
        num_bids = total_dice * DICE_SIDES

        # Own dice: one-hot encode each die
        my_dice = jax.lax.select(color == 0, state.dice[0], state.dice[1])
        # my_dice has shape (num_dice,) with values 1..6
        # One-hot: (num_dice, DICE_SIDES), then flatten
        dice_onehot = jax.nn.one_hot(my_dice - 1, DICE_SIDES).reshape(-1)

        bid_ids = jnp.arange(num_bids)
        bid_history = (bid_ids <= state.current_bid).astype(jnp.float32)
        # Zero out if no bid made yet
        bid_history = jnp.where(state.current_bid >= 0, bid_history, 0.0)

        return jnp.concatenate([dice_onehot, bid_history])


class LiarsDiceBidHistoryObservation(ArrayObservation):
    """Observation with explicit per-player bid tracking.

    Layout:
      [0 .. num_dice * DICE_SIDES - 1]:  One-hot encoding of own dice
      [next section]:  Current bid quantity (normalized by total_dice)
      [next]:  Current bid face (one-hot, DICE_SIDES)
      [next]:  Fraction of bid space remaining (how many bids are still legal)

    Total size = num_dice * DICE_SIDES + 1 + DICE_SIDES + 1
    """

    def __init__(self, num_dice: int = 5):
        self._num_dice = num_dice

    @property
    def shape(self) -> tuple[int, ...]:
        return (self._num_dice * DICE_SIDES + 1 + DICE_SIDES + 1,)

    @classmethod
    def from_state(cls, state, *, color: Array, num_dice: int, **_kwargs) -> Array:
        total_dice = 2 * num_dice

        # Own dice one-hot
        my_dice = jax.lax.select(color == 0, state.dice[0], state.dice[1])
        dice_onehot = jax.nn.one_hot(my_dice - 1, DICE_SIDES).reshape(-1)

        # Current bid info
        has_bid = (state.current_bid >= 0).astype(jnp.float32)
        bid_quantity = jnp.where(
            state.current_bid >= 0,
            (state.current_bid // DICE_SIDES + 1).astype(jnp.float32) / total_dice,
            0.0,
        )
        bid_face_idx = jnp.where(state.current_bid >= 0, state.current_bid % DICE_SIDES, 0)
        bid_face_onehot = jax.nn.one_hot(bid_face_idx, DICE_SIDES) * has_bid

        # Fraction of bid space remaining
        num_bids = total_dice * DICE_SIDES
        remaining = jnp.where(
            state.current_bid >= 0,
            (num_bids - state.current_bid - 1).astype(jnp.float32) / num_bids,
            1.0,
        )

        return jnp.concatenate(
            [
                dice_onehot,
                jnp.float32([bid_quantity]),
                bid_face_onehot,
                jnp.float32([remaining]),
            ]
        )


OBSERVATION_CLASSES: dict[str, type[Observation[Any]]] = {
    "default": LiarsDiceObservation,
    "compact": LiarsDiceBidHistoryObservation,
}
