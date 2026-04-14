"""Heuristic Quoridor baseline for evaluation.

Pure JAX implementation — JIT-compatible, works inside jax.lax.scan.
Strategy: advance toward opponent's base (row 8 in canonical view),
occasionally place horizontal walls near the opponent to block their path.
"""

from __future__ import annotations

import jax.numpy as jnp

from jaxpot.models.base import ModelOutput, PolicyValueModel

# In canonical (perspective-normalized) view:
#   - My pawn is near row 0, opponent near row 8
#   - N (action 1) = advance toward row 8 = toward goal
#   - S (action 0) = retreat toward row 0
#
# Move scores: advance > lateral > retreat
MOVE_SCORES = jnp.array(
    [
        -50.0,  # S(0) - retreat
        100.0,  # N(1) - advance toward goal
        10.0,  # E(2) - lateral
        10.0,  # W(3) - lateral
        -50.0,  # Jump S(4) - retreat 2 rows
        150.0,  # Jump N(5) - advance 2 rows (best)
        10.0,  # Jump E(6) - lateral jump
        10.0,  # Jump W(7) - lateral jump
        -25.0,  # SE(8)  - diagonal retreat
        -25.0,  # SW(9)  - diagonal retreat
        75.0,  # NE(10) - diagonal advance
        75.0,  # NW(11) - diagonal advance
    ],
    dtype=jnp.float32,
)

# Precompute wall grid coordinates (8x8 = 64 positions)
_wr = jnp.arange(8)
_wc = jnp.arange(8)
_WALL_ROWS, _WALL_COLS = jnp.meshgrid(_wr, _wc, indexing="ij")
WALL_ROWS_FLAT = _WALL_ROWS.reshape(64)  # (64,)
WALL_COLS_FLAT = _WALL_COLS.reshape(64)  # (64,)


class QuoridorBaseline(PolicyValueModel):
    """Heuristic Quoridor player.

    Accepts both observation formats:
      - Spatial (B, 9, 9, 4): original 4-channel board
      - Flat (B, 326): spatial_scalar with walls remaining appended

    Strategy:
      - Strongly prefer advancing north toward the goal (row 8).
      - Occasionally place horizontal walls near the opponent to block
        their path back toward row 0.
      - Vertical walls scored lower (block lateral movement, less useful).

    Returns policy_logits; value head returns zeros (unused).
    """

    def __init__(self, **kwargs):
        super().__init__(obs_shape=(326,), action_dim=140)

    @staticmethod
    def _extract_spatial(obs: jnp.ndarray) -> jnp.ndarray:
        """Extract spatial (B, 9, 9, 4) from either flat (B, 326) or spatial (B, 9, 9, 4) obs."""
        if obs.ndim == 2:
            # Flat obs: (B, 326) -> take first 324 and reshape
            return obs[:, :324].reshape(obs.shape[0], 9, 9, 4)
        # Already spatial: (B, 9, 9, 4)
        return obs

    def __call__(
        self,
        obs: jnp.ndarray,
        hidden_state: jnp.ndarray | None = None,
    ) -> ModelOutput:
        unbatched = obs.ndim in (1, 3)
        if unbatched:
            obs = obs[None]
        B = obs.shape[0]

        spatial = self._extract_spatial(obs)

        # ── Movement scores (actions 0-11) ──────────────────────────
        scores_move = jnp.broadcast_to(MOVE_SCORES, (B, 12))

        # ── Wall scores (actions 12-139) ────────────────────────────
        # Find opponent position from observation channel 1
        opp_plane = spatial[:, :, :, 1]  # (B, 9, 9)
        opp_flat = opp_plane.reshape(B, -1)  # (B, 81)
        opp_idx = jnp.argmax(opp_flat, axis=-1)  # (B,)
        opp_row = opp_idx // 9  # (B,)
        opp_col = opp_idx % 9  # (B,)

        # Manhattan distance from each wall position to opponent
        # Wall rows/cols: (64,), opponent: (B,) -> broadcast (B, 64)
        row_dist = jnp.abs(WALL_ROWS_FLAT[None, :] - opp_row[:, None])
        col_dist = jnp.abs(WALL_COLS_FLAT[None, :] - opp_col[:, None])
        manhattan = row_dist + col_dist  # (B, 64)

        # Horizontal walls: good for blocking N/S movement.
        # Bonus when wall is between the opponent and row 0 (their goal
        # in our canonical view) — i.e. wall_r < opp_row.
        h_blocking = jnp.where(WALL_ROWS_FLAT[None, :] < opp_row[:, None], 10.0, 0.0)
        h_wall_scores = 40.0 - 6.0 * manhattan + h_blocking  # (B, 64)

        # Vertical walls: less useful (block lateral movement).
        v_wall_scores = 15.0 - 6.0 * manhattan  # (B, 64)

        # ── Assemble full logit vector ──────────────────────────────
        scores = jnp.concatenate([scores_move, h_wall_scores, v_wall_scores], axis=-1)  # (B, 140)

        out = ModelOutput(
            value=jnp.zeros((B, 1), dtype=jnp.float32),
            policy_logits=scores,
        )
        if unbatched:
            out = ModelOutput(
                value=out.value[0],
                policy_logits=out.policy_logits[0],
            )
        return out
