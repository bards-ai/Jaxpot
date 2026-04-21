"""Heuristic Liar's Dice baseline for evaluation.

Pure JAX implementation — JIT-compatible, works inside jax.lax.scan.

Strategy:
- Count own dice for each face (including wilds).
- For each bid, estimate expected total matching dice (own + expected opponent).
- Call liar when the current bid exceeds the expected count by a threshold.
- When bidding, prefer faces with more own dice, bid conservatively.
"""

from __future__ import annotations

import jax.numpy as jnp

from jaxpot.models.base import ModelOutput, PolicyValueModel

DICE_SIDES = 6


class LiarsDiceBaseline(PolicyValueModel):
    """Heuristic Liar's Dice player.

    Observation layout (default obs):
      [0 : num_dice*6]                      — one-hot encoding of own dice
      [num_dice*6 : num_dice*6 + total*6]   — bid history flags

    Parametric: works for any num_dice value.
    """

    def __init__(self, num_dice: int = 5, **kwargs):
        self._num_dice = num_dice
        total_dice = 2 * num_dice
        num_bids = total_dice * DICE_SIDES
        obs_size = num_dice * DICE_SIDES + num_bids
        action_size = num_bids + 1

        super().__init__(obs_shape=(obs_size,), action_dim=action_size)
        # Store as plain ints for use in __call__
        self._total_dice = total_dice
        self._num_bids = num_bids
        self._obs_dice_size = num_dice * DICE_SIDES

    def __call__(
        self,
        obs: jnp.ndarray,
        hidden_state: jnp.ndarray | None = None,
    ) -> ModelOutput:
        B = obs.shape[0]
        num_dice = self._num_dice
        num_bids = self._num_bids
        obs_dice_size = self._obs_dice_size

        # --- Parse own dice from observation ---
        dice_onehot = obs[:, :obs_dice_size].reshape(B, num_dice, DICE_SIDES)
        face_counts = dice_onehot.sum(axis=1)  # (B, 6)

        # Wild face is index 5 (face value 6)
        wild_count = face_counts[:, 5]  # (B,)

        # Effective own count per face (non-wild faces get wilds added)
        own_effective = face_counts + wild_count[:, None]  # (B, 6)
        own_effective = own_effective.at[:, 5].set(wild_count)

        # --- Expected opponent count per face ---
        opp_expected = jnp.full((B, DICE_SIDES), num_dice * 2.0 / DICE_SIDES)
        opp_expected = opp_expected.at[:, 5].set(num_dice * 1.0 / DICE_SIDES)

        expected_total = own_effective + opp_expected  # (B, 6)

        # --- Score each bid action ---
        action_ids = jnp.arange(num_bids)
        bid_quantities = (action_ids // DICE_SIDES + 1).astype(jnp.float32)
        bid_face_idx = action_ids % DICE_SIDES

        expected_for_bid = expected_total[:, bid_face_idx]  # (B, num_bids)
        bid_scores = (expected_for_bid - bid_quantities[None, :]) * 3.0

        # --- Parse current bid from observation ---
        bid_history = obs[:, obs_dice_size:]  # (B, num_bids)
        has_any_bid = bid_history.sum(axis=-1) > 0  # (B,)
        current_bid = bid_history.sum(axis=-1).astype(jnp.int32) - 1  # (B,)

        # --- Score liar action ---
        safe_bid = jnp.clip(current_bid, 0, num_bids - 1)
        current_quantity = (safe_bid // DICE_SIDES + 1).astype(jnp.float32)
        current_face_idx = safe_bid % DICE_SIDES
        current_expected = expected_total[jnp.arange(B), current_face_idx]
        liar_score = (current_quantity - current_expected) * 3.0
        liar_score = jnp.where(has_any_bid, liar_score, -1e9)

        # --- Mask illegal bids ---
        legal_bids = action_ids[None, :] > current_bid[:, None]
        legal_bids = jnp.where(has_any_bid[:, None], legal_bids, True)
        bid_scores = jnp.where(legal_bids, bid_scores, -1e9)

        logits = jnp.concatenate([bid_scores, liar_score[:, None]], axis=-1)

        return ModelOutput(
            value=jnp.zeros((B, 1), dtype=jnp.float32),
            policy_logits=logits,
        )
