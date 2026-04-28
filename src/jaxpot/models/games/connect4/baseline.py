"""Heuristic Connect4 baseline for evaluation.

Pure JAX implementation — JIT-compatible, works inside jax.lax.scan.
Heuristic priority: Win > Block > Center preference.
"""

from __future__ import annotations

import jax.numpy as jnp

from jaxpot.models.base import ModelOutput, PolicyValueModel

# ---------------------------------------------------------------------------
# Precompute all 69 possible 4-in-a-row lines (as flat indices into 6×7 board)
# ---------------------------------------------------------------------------


def _generate_lines_flat() -> list[list[int]]:
    """Return flat indices (row*7+col) for every possible 4-in-a-row."""
    lines: list[list[int]] = []
    for r in range(6):
        for c in range(7):
            # Horizontal →
            if c + 3 < 7:
                lines.append([r * 7 + c + i for i in range(4)])
            # Vertical ↓
            if r + 3 < 6:
                lines.append([(r + i) * 7 + c for i in range(4)])
            # Diagonal ↘
            if r + 3 < 6 and c + 3 < 7:
                lines.append([(r + i) * 7 + (c + i) for i in range(4)])
            # Diagonal ↗
            if r >= 3 and c + 3 < 7:
                lines.append([(r - i) * 7 + (c + i) for i in range(4)])
    return lines


ALL_LINES = jnp.array(_generate_lines_flat(), dtype=jnp.int32)  # (69, 4)

# Center preference scores per column (0–6): center=3 highest
CENTER_SCORES = jnp.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0])


class Connect4Baseline(PolicyValueModel):
    """Heuristic Connect4 player.

    Observation shape: (B, 6, 7, 2)
        channel 0 = current player's pieces
        channel 1 = opponent's pieces

    Returns policy_logits scored by: win(1000) > block(100) > center(0–3).
    Value head returns zeros (unused for opponent evaluation).
    """

    def __init__(self, **kwargs):
        super().__init__(obs_shape=(6, 7, 2), action_dim=7)

    def __call__(
        self,
        obs: jnp.ndarray,
        hidden_state: jnp.ndarray | None = None,
    ) -> ModelOutput:
        B = obs.shape[0]
        my_pieces = obs[:, :, :, 0]  # (B, 6, 7)
        opp_pieces = obs[:, :, :, 1]  # (B, 6, 7)
        occupied = my_pieces + opp_pieces

        # Landing row per column: 5 - #pieces_in_column
        col_counts = occupied.sum(axis=1).astype(jnp.int32)  # (B, 7)
        landing_rows = 5 - col_counts  # (B, 7)
        valid_cols = landing_rows >= 0  # (B, 7)

        # One-hot board for the newly placed piece, per column: (B, 7, 6, 7)
        rows = jnp.arange(6)[None, None, :, None]  # (1, 1, 6, 1)
        cols = jnp.arange(7)[None, None, None, :]  # (1, 1, 1, 7)
        target_row = landing_rows[:, :, None, None]  # (B, 7, 1, 1)
        target_col = jnp.arange(7)[None, :, None, None]  # (1, 7, 1, 1)

        new_piece = ((rows == target_row) & (cols == target_col)).astype(jnp.float32)

        # Simulated boards after placing in each column: (B, 7, 6, 7)
        my_sim = my_pieces[:, None, :, :] + new_piece
        opp_sim = opp_pieces[:, None, :, :] + new_piece

        # Flatten spatial dims → (B, 7, 42), then gather line cells
        my_line_sums = my_sim.reshape(B, 7, 42)[:, :, ALL_LINES].sum(axis=-1)  # (B, 7, 69)
        opp_line_sums = opp_sim.reshape(B, 7, 42)[:, :, ALL_LINES].sum(axis=-1)  # (B, 7, 69)

        can_win = (my_line_sums == 4).any(axis=-1)  # (B, 7)
        can_block = (opp_line_sums == 4).any(axis=-1)  # (B, 7)

        scores = (
            can_win.astype(jnp.float32) * 1000.0
            + can_block.astype(jnp.float32) * 100.0
            + CENTER_SCORES[None, :]
        )
        scores = jnp.where(valid_cols, scores, -1e9)

        return ModelOutput(
            value=jnp.zeros((B, 1), dtype=jnp.float32),
            policy_logits=scores,
        )
