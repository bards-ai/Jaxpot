"""BFS shortest-path Quoridor baseline for evaluation.

Pure JAX implementation — JIT-compatible, works inside jax.lax.scan.

Strategy (based on competitive Quoridor AI research):
  - Movement: always follow the BFS shortest path to goal row.
  - Walls: place walls that block cells on the opponent's shortest path.
  - When ahead in path distance, prefer moving; when behind, prefer walling.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxpot.models.base import ModelOutput, PolicyValueModel

BOARD_SIZE = 9
WALL_SIZE = 8


def _compute_move_masks(h_walls, v_walls):
    """Compute can_n/s/e/w [9,9] bool arrays from wall layout [8,8].

    Includes boundary constraints (can't move off the board).
    """
    h_pad = jnp.zeros((9, 9), dtype=jnp.bool_).at[:8, :8].set(h_walls > 0.5)
    v_pad = jnp.zeros((9, 9), dtype=jnp.bool_).at[:8, :8].set(v_walls > 0.5)

    # North blocked: h_walls[r, c] or h_walls[r, c-1]
    hw_r = h_pad
    hw_l = jnp.zeros((9, 9), dtype=jnp.bool_).at[:, 1:].set(h_pad[:, :8])
    n_blocked = hw_r | hw_l
    can_n = (~n_blocked).at[8, :].set(False)

    # South blocked: shift n_blocked down one row
    s_blocked = jnp.zeros((9, 9), dtype=jnp.bool_).at[1:, :].set(n_blocked[:8, :])
    can_s = (~s_blocked).at[0, :].set(False)

    # East blocked: v_walls[r, c] or v_walls[r-1, c]
    vw_d = v_pad
    vw_u = jnp.zeros((9, 9), dtype=jnp.bool_).at[1:, :].set(v_pad[:8, :])
    e_blocked = vw_d | vw_u
    can_e = (~e_blocked).at[:, 8].set(False)

    # West blocked: shift e_blocked right one col
    w_blocked = jnp.zeros((9, 9), dtype=jnp.bool_).at[:, 1:].set(e_blocked[:, :8])
    can_w = (~w_blocked).at[:, 0].set(False)

    return can_n, can_s, can_e, can_w


def _bfs_from_row(goal_row, can_n, can_s, can_e, can_w):
    """BFS distance from each cell to a goal row.

    Uses wavefront expansion: dist[r,c] = min steps to reach any cell in goal_row.
    All inputs [9,9]. Returns [9,9] float (99.0 = unreachable).
    """
    dist = jnp.full((9, 9), 99.0).at[goal_row, :].set(0.0)

    def step(dist, _):
        # For cell (r,c), if I can move in direction d, my dist <= neighbor_dist + 1
        # North neighbor: dist[r+1, c] (move north to reach it)
        d_n = jnp.full((9, 9), 99.0).at[:8, :].set(dist[1:, :])
        # South neighbor: dist[r-1, c]
        d_s = jnp.full((9, 9), 99.0).at[1:, :].set(dist[:8, :])
        # East neighbor: dist[r, c+1]
        d_e = jnp.full((9, 9), 99.0).at[:, :8].set(dist[:, 1:])
        # West neighbor: dist[r, c-1]
        d_w = jnp.full((9, 9), 99.0).at[:, 1:].set(dist[:, :8])

        cand = jnp.minimum(
            jnp.minimum(
                jnp.where(can_n, d_n + 1, 99.0),
                jnp.where(can_s, d_s + 1, 99.0),
            ),
            jnp.minimum(
                jnp.where(can_e, d_e + 1, 99.0),
                jnp.where(can_w, d_w + 1, 99.0),
            ),
        )
        return jnp.minimum(dist, cand), None

    dist, _ = jax.lax.scan(step, dist, None, length=50)
    return dist


def _bfs_from_cell(start_r, start_c, can_n, can_s, can_e, can_w):
    """BFS distance from a specific cell to all others. Returns [9,9] float."""
    dist = jnp.full((9, 9), 99.0).at[start_r, start_c].set(0.0)

    def step(dist, _):
        d_n = jnp.full((9, 9), 99.0).at[:8, :].set(dist[1:, :])
        d_s = jnp.full((9, 9), 99.0).at[1:, :].set(dist[:8, :])
        d_e = jnp.full((9, 9), 99.0).at[:, :8].set(dist[:, 1:])
        d_w = jnp.full((9, 9), 99.0).at[:, 1:].set(dist[:, :8])

        cand = jnp.minimum(
            jnp.minimum(
                jnp.where(can_n, d_n + 1, 99.0),
                jnp.where(can_s, d_s + 1, 99.0),
            ),
            jnp.minimum(
                jnp.where(can_e, d_e + 1, 99.0),
                jnp.where(can_w, d_w + 1, 99.0),
            ),
        )
        return jnp.minimum(dist, cand), None

    dist, _ = jax.lax.scan(step, dist, None, length=50)
    return dist


class QuoridorBFSBaseline(PolicyValueModel):
    """BFS shortest-path baseline for Quoridor.

    Much stronger than the simple heuristic baseline. Uses BFS distance maps
    to follow the shortest path and place walls on the opponent's path.
    """

    def __init__(self, **kwargs):
        super().__init__(obs_shape=(326,), action_dim=140)

    def __call__(self, obs, hidden_state=None):
        unbatched = obs.ndim in (1, 3)
        if unbatched:
            obs = obs[None]
        B = obs.shape[0]

        scores = jax.vmap(self._score_single)(obs)

        out = ModelOutput(
            value=jnp.zeros((B, 1), dtype=jnp.float32),
            policy_logits=scores,
        )
        if unbatched:
            out = ModelOutput(value=out.value[0], policy_logits=out.policy_logits[0])
        return out

    @staticmethod
    def _score_single(obs):
        """Compute policy logits for a single observation."""
        # ── Parse observation ──
        spatial = obs[:324].reshape(9, 9, 4)
        my_walls_frac = obs[324]

        my_plane = spatial[:, :, 0]
        opp_plane = spatial[:, :, 1]
        h_walls = spatial[:8, :8, 2]
        v_walls = spatial[:8, :8, 3]

        my_flat = jnp.argmax(my_plane.flatten())
        opp_flat = jnp.argmax(opp_plane.flatten())
        my_r, my_c = my_flat // 9, my_flat % 9
        opp_r, opp_c = opp_flat // 9, opp_flat % 9

        # ── BFS distance maps ──
        can_n, can_s, can_e, can_w = _compute_move_masks(h_walls, v_walls)
        # My goal is row 8 (canonical: I start near row 0, advance toward row 8)
        my_goal_dist = _bfs_from_row(8, can_n, can_s, can_e, can_w)
        # Opponent's goal is row 0
        opp_goal_dist = _bfs_from_row(0, can_n, can_s, can_e, can_w)
        # BFS from opponent's position (to identify cells on their shortest path)
        opp_from_pos = _bfs_from_cell(opp_r, opp_c, can_n, can_s, can_e, can_w)

        my_dist = my_goal_dist[my_r, my_c]
        opp_dist = opp_goal_dist[opp_r, opp_c]

        # ── Movement scores (actions 0-11) ──
        # Score = improvement in distance to goal (positive = getting closer)
        dr = jnp.array([-1, 1, 0, 0], dtype=jnp.int32)
        dc = jnp.array([0, 0, 1, -1], dtype=jnp.int32)

        def _target_dist(r, c):
            r, c = jnp.clip(r, 0, 8), jnp.clip(c, 0, 8)
            return my_goal_dist[r, c]

        # Cardinal (0-3): S, N, E, W
        cardinal = jnp.array([my_dist - _target_dist(my_r + dr[d], my_c + dc[d]) for d in range(4)])

        # Jump (4-7): 2 cells in direction d (over opponent)
        jump = jnp.array(
            [my_dist - _target_dist(my_r + 2 * dr[d], my_c + 2 * dc[d]) for d in range(4)]
        )

        # Diagonal (8-11): SE, SW, NE, NW — land near opponent
        diag_dr = jnp.array([-1, -1, 1, 1], dtype=jnp.int32)
        diag_dc = jnp.array([1, -1, 1, -1], dtype=jnp.int32)
        diagonal = jnp.array(
            [my_dist - _target_dist(opp_r + diag_dr[d], opp_c + diag_dc[d]) for d in range(4)]
        )

        # Scale improvements to logits; jumps get small bonus (2 cells for 1 action)
        move_scores = jnp.concatenate(
            [
                cardinal * 100.0,
                jump * 100.0 + 5.0,
                diagonal * 100.0,
            ]
        )

        # ── Wall scores (actions 12-139) ──
        wall_idx = jnp.arange(64)
        wall_r = wall_idx // 8
        wall_c = wall_idx % 8

        # Identify cells on opponent's shortest path:
        # Cell is on shortest path if dist_from_opp + dist_to_goal == opp_shortest_dist
        on_opp_path = (opp_from_pos + opp_goal_dist) <= (opp_dist + 0.5)  # [9,9] bool

        # --- Horizontal walls (actions 12-75) ---
        # h_wall at (r, c) blocks N from (r,c)&(r,c+1) and S from (r+1,c)&(r+1,c+1)
        # Effective if any of the 4 affected cells are on opponent's shortest path
        wc1 = jnp.clip(wall_c + 1, 0, 8)
        wr1 = jnp.clip(wall_r + 1, 0, 8)
        h_on_path = (
            on_opp_path[wall_r, wall_c]
            | on_opp_path[wall_r, wc1]
            | on_opp_path[wr1, wall_c]
            | on_opp_path[wr1, wc1]
        )
        # Bonus when wall is between opponent and their goal (row 0)
        h_between = (wall_r < opp_r).astype(jnp.float32)
        # Distance to opponent for proximity scoring
        opp_manhattan = (jnp.abs(wall_r - opp_r) + jnp.abs(wall_c - opp_c)).astype(jnp.float32)

        h_wall_scores = (
            h_on_path.astype(jnp.float32) * 40.0  # big bonus for blocking opponent's path
            + h_between * 15.0  # bonus for being between opp and goal
            - opp_manhattan * 2.0  # prefer closer to opponent
        )

        # --- Vertical walls (actions 76-139) ---
        v_on_path = (
            on_opp_path[wall_r, wall_c]
            | on_opp_path[wr1, wall_c]
            | on_opp_path[wall_r, wc1]
            | on_opp_path[wr1, wc1]
        )
        v_wall_scores = (
            v_on_path.astype(jnp.float32) * 25.0  # vertical less effective for N/S blocking
            - opp_manhattan * 2.0
        )

        # ── Strategic modulation ──
        path_diff = opp_dist - my_dist  # positive = I'm ahead

        # When ahead: run to goal (big move bonus, suppress walls)
        # When behind: place walls (big wall bonus, suppress moves)
        # When even: slight preference for moves
        move_bonus = jnp.where(
            path_diff >= 0,
            50.0 + jnp.clip(path_diff * 15.0, 0.0, 60.0),  # ahead: strong move
            jnp.clip(30.0 + path_diff * 20.0, -50.0, 30.0),  # behind: weaken moves
        )
        wall_desire = jnp.where(
            path_diff >= 0,
            jnp.clip(-path_diff * 10.0, -60.0, 0.0),  # ahead: suppress walls
            jnp.clip(-path_diff * 30.0, 0.0, 80.0),  # behind: strong wall desire
        )

        # Conserve walls when few remaining (< 2 walls left)
        wall_penalty = jnp.where(my_walls_frac < 0.2, -60.0, 0.0)

        scores = jnp.concatenate(
            [
                move_scores + move_bonus,
                h_wall_scores + wall_desire + wall_penalty,
                v_wall_scores + wall_desire + wall_penalty,
            ]
        )

        return scores
