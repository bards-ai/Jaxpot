"""Pure JAX game logic for Quoridor.

Board is 9x9. Two players each start with 10 walls.
Player 0 starts at (0, 4), goal is row 8.
Player 1 starts at (8, 4), goal is row 0.

Action space (140 total):
  0-3:   Cardinal moves S(0) N(1) E(2) W(3)
  4-7:   Straight jumps over opponent
  8-11:  Diagonal jumps (when straight jump blocked by wall)
  12-75: Horizontal wall at (r, c) = action 12 + r*8 + c
  76-139: Vertical wall at (r, c) = action 76 + r*8 + c

Wall representation:
  h_walls[r, c]: horizontal wall between rows r and r+1, spanning cols c and c+1
  v_walls[r, c]: vertical wall between cols c and c+1, spanning rows r and r+1
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

# Direction offsets for S, N, E, W (row_delta, col_delta)
DR = jnp.array([-1, 1, 0, 0], dtype=jnp.int32)
DC = jnp.array([0, 0, 1, -1], dtype=jnp.int32)

BOARD_SIZE = 9
WALL_SIZE = 8  # 8x8 grid of wall positions
NUM_ACTIONS = 140
NUM_WALLS_PER_PLAYER = 10
DEFAULT_MAX_STEPS = 200  # Max total moves before game is truncated


class GameState(NamedTuple):
    """Internal game state for Quoridor."""

    color: Array = jnp.int32(0)  # whose turn (0 or 1)
    pawn_pos: Array = jnp.array([4, 76], dtype=jnp.int32)  # flat 9x9 indices
    walls_remaining: Array = jnp.array(
        [NUM_WALLS_PER_PLAYER, NUM_WALLS_PER_PLAYER], dtype=jnp.int32
    )
    h_walls: Array = jnp.zeros((WALL_SIZE, WALL_SIZE), dtype=jnp.bool_)
    v_walls: Array = jnp.zeros((WALL_SIZE, WALL_SIZE), dtype=jnp.bool_)
    winner: Array = jnp.int32(-1)  # -1=ongoing, 0/1=winner
    h_wall_owners: Array = jnp.full(
        (WALL_SIZE, WALL_SIZE), -1, dtype=jnp.int8
    )  # -1=none, 0/1=player
    v_wall_owners: Array = jnp.full(
        (WALL_SIZE, WALL_SIZE), -1, dtype=jnp.int8
    )  # -1=none, 0/1=player
    # Cached per-cell movement masks (only depend on wall positions)
    can_n: Array = jnp.ones(81, dtype=jnp.bool_)
    can_s: Array = jnp.ones(81, dtype=jnp.bool_)
    can_e: Array = jnp.ones(81, dtype=jnp.bool_)
    can_w: Array = jnp.ones(81, dtype=jnp.bool_)


def _pos_to_rc(pos: Array) -> tuple[Array, Array]:
    """Convert flat index to (row, col)."""
    return pos // BOARD_SIZE, pos % BOARD_SIZE


def _rc_to_pos(r: Array, c: Array) -> Array:
    """Convert (row, col) to flat index."""
    return r * BOARD_SIZE + c


def _precompute_move_masks(h_walls: Array, v_walls: Array):
    """Pre-compute per-cell movement masks using array ops (no vmap).

    Returns can_n, can_s, can_e, can_w as flat [81] boolean arrays.
    can_X[i] = True means cell i can move in direction X without hitting a wall.
    (Board boundary checks are NOT included — caller must handle those.)
    """
    # h_walls[r, c] blocks movement between row r and row r+1 at columns c, c+1
    # v_walls[r, c] blocks movement between col c and col c+1 at rows r, r+1

    # Pad h_walls to 9x9 so indexing is safe, then check both relevant wall positions
    # For cell (r, c) moving North (r→r+1): blocked by h_walls[r, c] or h_walls[r, c-1]
    #   h_walls[r, c] exists when r < 8 and c < 8
    #   h_walls[r, c-1] exists when r < 8 and c > 0
    h_pad = jnp.zeros((BOARD_SIZE, BOARD_SIZE), dtype=jnp.bool_)
    h_pad = h_pad.at[:WALL_SIZE, :WALL_SIZE].set(h_walls)  # [9, 9] with padding

    # For North (increasing row): cell (r,c) blocked by h_walls[r,c] | h_walls[r,c-1]
    hw_right = h_pad  # h_walls[r, c] — already aligned
    hw_left = jnp.zeros((BOARD_SIZE, BOARD_SIZE), dtype=jnp.bool_)
    hw_left = hw_left.at[:, 1:].set(h_pad[:, : BOARD_SIZE - 1])  # h_walls[r, c-1] shifted right

    n_blocked_2d = hw_right | hw_left  # [9, 9]
    # Row 8 can never go north (boundary), but we exclude boundary checks here
    # h_walls only valid for r < 8, which is automatic since h_pad[8, :] = 0
    can_n = ~n_blocked_2d.flatten()  # can move north from cell i

    # For South (decreasing row): cell (r,c) blocked by h_walls[r-1,c] | h_walls[r-1,c-1]
    s_blocked_2d = jnp.zeros((BOARD_SIZE, BOARD_SIZE), dtype=jnp.bool_)
    s_blocked_2d = s_blocked_2d.at[1:, :].set(
        n_blocked_2d[: BOARD_SIZE - 1, :]
    )  # shift down by 1 row
    can_s = ~s_blocked_2d.flatten()

    # Similarly for vertical walls
    v_pad = jnp.zeros((BOARD_SIZE, BOARD_SIZE), dtype=jnp.bool_)
    v_pad = v_pad.at[:WALL_SIZE, :WALL_SIZE].set(v_walls)

    # For East (increasing col): cell (r,c) blocked by v_walls[r,c] | v_walls[r-1,c]
    vw_down = v_pad  # v_walls[r, c]
    vw_up = jnp.zeros((BOARD_SIZE, BOARD_SIZE), dtype=jnp.bool_)
    vw_up = vw_up.at[1:, :].set(v_pad[: BOARD_SIZE - 1, :])  # v_walls[r-1, c] shifted down

    e_blocked_2d = vw_down | vw_up
    can_e = ~e_blocked_2d.flatten()

    # For West (decreasing col): cell (r,c) blocked by v_walls[r,c-1] | v_walls[r-1,c-1]
    w_blocked_2d = jnp.zeros((BOARD_SIZE, BOARD_SIZE), dtype=jnp.bool_)
    w_blocked_2d = w_blocked_2d.at[:, 1:].set(
        e_blocked_2d[:, : BOARD_SIZE - 1]
    )  # shift right by 1 col
    can_w = ~w_blocked_2d.flatten()

    return can_n, can_s, can_e, can_w


# ── Bitwise BFS constants ──────────────────────────────────────────────────
# Pack 81-cell board into (a: int32 bits 0-31, b: int32 bits 0-31, c: int32 bits 0-16).
# Cell i maps to: a[i] if i<32, b[i-32] if 32<=i<64, c[i-64] if 64<=i<81.

_VALID_C = jnp.int32((1 << 17) - 1)

# Bit packing helpers
_BIT_A = jnp.array([np.int32(1) << np.int32(i) for i in range(32)], dtype=jnp.int32)
_BIT_B = jnp.array([np.int32(1) << np.int32(i) for i in range(32)], dtype=jnp.int32)
_BIT_C = jnp.array([np.int32(1) << np.int32(i) for i in range(17)], dtype=jnp.int32)


def _pack_np(arr_81):
    """Pack 81-element bool array into (a, b, c) int32 triple using numpy."""
    a = np.int32(0)
    for i in range(32):
        if arr_81[i]:
            a |= np.int32(1) << np.int32(i)
    b = np.int32(0)
    for i in range(32):
        if arr_81[32 + i]:
            b |= np.int32(1) << np.int32(i)
    c = np.int32(0)
    for i in range(17):
        if arr_81[64 + i]:
            c |= np.int32(1) << np.int32(i)
    return a, b, c


# Boundary bitmasks (which cells can move in each direction)
_rows_np = np.arange(81) // 9
_cols_np = np.arange(81) % 9
_BOUND_N_A, _BOUND_N_B, _BOUND_N_C = _pack_np(_rows_np < 8)
_BOUND_S_A, _BOUND_S_B, _BOUND_S_C = _pack_np(_rows_np > 0)
_BOUND_E_A, _BOUND_E_B, _BOUND_E_C = _pack_np(_cols_np < 8)
_BOUND_W_A, _BOUND_W_B, _BOUND_W_C = _pack_np(_cols_np > 0)

_BOUND_N_A = jnp.int32(_BOUND_N_A)
_BOUND_N_B = jnp.int32(_BOUND_N_B)
_BOUND_N_C = jnp.int32(_BOUND_N_C)
_BOUND_S_A = jnp.int32(_BOUND_S_A)
_BOUND_S_B = jnp.int32(_BOUND_S_B)
_BOUND_S_C = jnp.int32(_BOUND_S_C)
_BOUND_E_A = jnp.int32(_BOUND_E_A)
_BOUND_E_B = jnp.int32(_BOUND_E_B)
_BOUND_E_C = jnp.int32(_BOUND_E_C)
_BOUND_W_A = jnp.int32(_BOUND_W_A)
_BOUND_W_B = jnp.int32(_BOUND_W_B)
_BOUND_W_C = jnp.int32(_BOUND_W_C)

# Goal row bitmasks
# Player 0 goal: row 8 (cells 72-80)
_goal0_np = np.zeros(81, dtype=bool)
_goal0_np[72:81] = True
_GOAL0_A, _GOAL0_B, _GOAL0_C = _pack_np(_goal0_np)
_GOAL0_A = jnp.int32(_GOAL0_A)
_GOAL0_B = jnp.int32(_GOAL0_B)
_GOAL0_C = jnp.int32(_GOAL0_C)

# Player 1 goal: row 0 (cells 0-8)
_goal1_np = np.zeros(81, dtype=bool)
_goal1_np[0:9] = True
_GOAL1_A, _GOAL1_B, _GOAL1_C = _pack_np(_goal1_np)
_GOAL1_A = jnp.int32(_GOAL1_A)
_GOAL1_B = jnp.int32(_GOAL1_B)
_GOAL1_C = jnp.int32(_GOAL1_C)


# Wall disable bitmasks: for each of 64 wall positions, which bits to clear.
def _build_disable_masks():
    """Build [64] arrays of (a, b, c) disable bitmasks for each wall position."""
    names = ["h_n", "h_s", "v_e", "v_w"]
    result = {}
    for name in names:
        result[f"{name}_a"] = np.zeros(64, dtype=np.int32)
        result[f"{name}_b"] = np.zeros(64, dtype=np.int32)
        result[f"{name}_c"] = np.zeros(64, dtype=np.int32)

    def set_bit(arr_a, arr_b, arr_c, idx, cell):
        if cell < 32:
            arr_a[idx] |= np.int32(1) << np.int32(cell)
        elif cell < 64:
            arr_b[idx] |= np.int32(1) << np.int32(cell - 32)
        else:
            arr_c[idx] |= np.int32(1) << np.int32(cell - 64)

    for idx in range(64):
        r, c = idx // 8, idx % 8
        # Horizontal wall: blocks can_n at (r*9+c, r*9+c+1), can_s at ((r+1)*9+c, (r+1)*9+c+1)
        for cell in [r * 9 + c, r * 9 + c + 1]:
            set_bit(result["h_n_a"], result["h_n_b"], result["h_n_c"], idx, cell)
        for cell in [(r + 1) * 9 + c, (r + 1) * 9 + c + 1]:
            set_bit(result["h_s_a"], result["h_s_b"], result["h_s_c"], idx, cell)
        # Vertical wall: blocks can_e at (r*9+c, (r+1)*9+c), can_w at (r*9+c+1, (r+1)*9+c+1)
        for cell in [r * 9 + c, (r + 1) * 9 + c]:
            set_bit(result["v_e_a"], result["v_e_b"], result["v_e_c"], idx, cell)
        for cell in [r * 9 + c + 1, (r + 1) * 9 + c + 1]:
            set_bit(result["v_w_a"], result["v_w_b"], result["v_w_c"], idx, cell)
    return result


_dis = _build_disable_masks()
_H_DISABLE_N_A = jnp.array(_dis["h_n_a"])
_H_DISABLE_N_B = jnp.array(_dis["h_n_b"])
_H_DISABLE_N_C = jnp.array(_dis["h_n_c"])
_H_DISABLE_S_A = jnp.array(_dis["h_s_a"])
_H_DISABLE_S_B = jnp.array(_dis["h_s_b"])
_H_DISABLE_S_C = jnp.array(_dis["h_s_c"])
_V_DISABLE_E_A = jnp.array(_dis["v_e_a"])
_V_DISABLE_E_B = jnp.array(_dis["v_e_b"])
_V_DISABLE_E_C = jnp.array(_dis["v_e_c"])
_V_DISABLE_W_A = jnp.array(_dis["v_w_a"])
_V_DISABLE_W_B = jnp.array(_dis["v_w_b"])
_V_DISABLE_W_C = jnp.array(_dis["v_w_c"])

del _rows_np, _cols_np, _goal0_np, _goal1_np, _dis


def _pack(arr_81: Array) -> tuple[Array, Array, Array]:
    """Pack a [81] boolean array into an (int32, int32, int32) triple."""
    a = jnp.sum(jnp.where(arr_81[:32], _BIT_A, jnp.int32(0)))
    b = jnp.sum(jnp.where(arr_81[32:64], _BIT_B, jnp.int32(0)))
    c = jnp.sum(jnp.where(arr_81[64:81], _BIT_C, jnp.int32(0)))
    return a, b, c


def _expand_frontier_bits(
    f_a,
    f_b,
    f_c,
    v_a,
    v_b,
    v_c,
    cn_a,
    cn_b,
    cn_c,
    cs_a,
    cs_b,
    cs_c,
    ce_a,
    ce_b,
    ce_c,
    cw_a,
    cw_b,
    cw_c,
):
    """Bitwise BFS expansion on (a, b, c) packed board representation.

    Board cells 0-80 packed as: a=bits[0:32], b=bits[32:64], c=bits[64:81].
    North=+9, South=-9, East=+1, West=-1 in flat cell index.
    """
    srl = jax.lax.shift_right_logical

    # North (+9): shift left by 9 across (a, b, c)
    ma = f_a & _BOUND_N_A & cn_a
    mb = f_b & _BOUND_N_B & cn_b
    mc = f_c & _BOUND_N_C & cn_c
    n_a = ma << 9
    n_b = (mb << 9) | srl(ma, jnp.int32(23))
    n_c = (mc << 9) | srl(mb, jnp.int32(23))

    # South (-9): shift right by 9
    ma = f_a & _BOUND_S_A & cs_a
    mb = f_b & _BOUND_S_B & cs_b
    mc = f_c & _BOUND_S_C & cs_c
    s_a = srl(ma, jnp.int32(9)) | (mb << 23)
    s_b = srl(mb, jnp.int32(9)) | (mc << 23)
    s_c = srl(mc, jnp.int32(9))

    # East (+1): shift left by 1
    ma = f_a & _BOUND_E_A & ce_a
    mb = f_b & _BOUND_E_B & ce_b
    mc = f_c & _BOUND_E_C & ce_c
    e_a = ma << 1
    e_b = (mb << 1) | srl(ma, jnp.int32(31))
    e_c = (mc << 1) | srl(mb, jnp.int32(31))

    # West (-1): shift right by 1
    ma = f_a & _BOUND_W_A & cw_a
    mb = f_b & _BOUND_W_B & cw_b
    mc = f_c & _BOUND_W_C & cw_c
    w_a = srl(ma, jnp.int32(1)) | (mb << 31)
    w_b = srl(mb, jnp.int32(1)) | (mc << 31)
    w_c = srl(mc, jnp.int32(1))

    # Combine, mask with ~visited, apply valid mask on c
    new_a = (n_a | s_a | e_a | w_a) & ~v_a
    new_b = (n_b | s_b | e_b | w_b) & ~v_b
    new_c = (n_c | s_c | e_c | w_c) & ~v_c & _VALID_C

    return v_a | new_a, v_b | new_b, v_c | new_c, new_a, new_b, new_c


def _has_path_both_bits(
    pawn0, pawn1, cn_a, cn_b, cn_c, cs_a, cs_b, cs_c, ce_a, ce_b, ce_c, cw_a, cw_b, cw_c
):
    """Two-player BFS using bitwise ops. Returns True iff both players have paths."""

    def _init_bit(pos):
        a = jnp.where(pos < 32, jnp.int32(1) << jnp.clip(pos, 0, 31), jnp.int32(0))
        b = jnp.where(
            (pos >= 32) & (pos < 64), jnp.int32(1) << jnp.clip(pos - 32, 0, 31), jnp.int32(0)
        )
        c = jnp.where(pos >= 64, jnp.int32(1) << jnp.clip(pos - 64, 0, 16), jnp.int32(0))
        return a, b, c

    f0_a, f0_b, f0_c = _init_bit(pawn0)
    f1_a, f1_b, f1_c = _init_bit(pawn1)
    v0_a, v0_b, v0_c = f0_a, f0_b, f0_c
    v1_a, v1_b, v1_c = f1_a, f1_b, f1_c

    def bfs_step(carry, _):
        (v0_a, v0_b, v0_c, f0_a, f0_b, f0_c, v1_a, v1_b, v1_c, f1_a, f1_b, f1_c) = carry
        v0_a, v0_b, v0_c, f0_a, f0_b, f0_c = _expand_frontier_bits(
            f0_a,
            f0_b,
            f0_c,
            v0_a,
            v0_b,
            v0_c,
            cn_a,
            cn_b,
            cn_c,
            cs_a,
            cs_b,
            cs_c,
            ce_a,
            ce_b,
            ce_c,
            cw_a,
            cw_b,
            cw_c,
        )
        v1_a, v1_b, v1_c, f1_a, f1_b, f1_c = _expand_frontier_bits(
            f1_a,
            f1_b,
            f1_c,
            v1_a,
            v1_b,
            v1_c,
            cn_a,
            cn_b,
            cn_c,
            cs_a,
            cs_b,
            cs_c,
            ce_a,
            ce_b,
            ce_c,
            cw_a,
            cw_b,
            cw_c,
        )
        return (v0_a, v0_b, v0_c, f0_a, f0_b, f0_c, v1_a, v1_b, v1_c, f1_a, f1_b, f1_c), None

    carry, _ = jax.lax.scan(
        bfs_step,
        (v0_a, v0_b, v0_c, f0_a, f0_b, f0_c, v1_a, v1_b, v1_c, f1_a, f1_b, f1_c),
        None,
        length=48,
    )
    v0_a, v0_b, v0_c, _, _, _, v1_a, v1_b, v1_c, _, _, _ = carry

    # Player 0 goal: row 8 (cells 72-80)
    p0_ok = (v0_a & _GOAL0_A) | (v0_b & _GOAL0_B) | (v0_c & _GOAL0_C)
    # Player 1 goal: row 0 (cells 0-8)
    p1_ok = (v1_a & _GOAL1_A) | (v1_b & _GOAL1_B) | (v1_c & _GOAL1_C)
    return (p0_ok != 0) & (p1_ok != 0)


def _wall_legal_mask(
    state: GameState, base_can_n: Array, base_can_s: Array, base_can_e: Array, base_can_w: Array
) -> Array:
    """Compute legality mask for all 128 wall placements using bitwise BFS.

    Returns [128] bool array: first 64 horizontal, next 64 vertical.
    Accepts pre-computed move masks to avoid redundant _precompute_move_masks call.
    """
    h_walls = state.h_walls
    v_walls = state.v_walls
    has_walls = state.walls_remaining[state.color] > 0

    # All 64 wall positions as (r, c) pairs
    wall_r = jnp.arange(64) // WALL_SIZE
    wall_c = jnp.arange(64) % WALL_SIZE

    # --- Overlap checks (vectorized, no vmap) ---
    h_flat = h_walls.flatten()  # [64]
    v_flat = v_walls.flatten()  # [64]

    # Horizontal: not placed, no left/right overlap, no cross
    h_not_placed = ~h_flat
    h_no_left = jnp.where(
        wall_c > 0, ~h_walls[wall_r, jnp.clip(wall_c - 1, 0, WALL_SIZE - 1)], True
    )
    h_no_right = jnp.where(
        wall_c < WALL_SIZE - 1, ~h_walls[wall_r, jnp.clip(wall_c + 1, 0, WALL_SIZE - 1)], True
    )
    h_no_cross = ~v_flat
    h_placement_ok = h_not_placed & h_no_left & h_no_right & h_no_cross & has_walls

    # Vertical: not placed, no up/down overlap, no cross
    v_not_placed = ~v_flat
    v_no_up = jnp.where(wall_r > 0, ~v_walls[jnp.clip(wall_r - 1, 0, WALL_SIZE - 1), wall_c], True)
    v_no_down = jnp.where(
        wall_r < WALL_SIZE - 1, ~v_walls[jnp.clip(wall_r + 1, 0, WALL_SIZE - 1), wall_c], True
    )
    v_no_cross = ~h_flat
    v_placement_ok = v_not_placed & v_no_up & v_no_down & v_no_cross & has_walls

    # --- Pack base masks into bitmasks once ---
    cn_a, cn_b, cn_c = _pack(base_can_n)
    cs_a, cs_b, cs_c = _pack(base_can_s)
    ce_a, ce_b, ce_c = _pack(base_can_e)
    cw_a, cw_b, cw_c = _pack(base_can_w)

    # --- Bitwise BFS path checks ---
    pawn0 = state.pawn_pos[0]
    pawn1 = state.pawn_pos[1]

    # Horizontal walls: modify can_n and can_s bitmasks
    # When placement_ok is False, pass unmodified masks (BFS trivially succeeds,
    # result is ANDed with False anyway)
    h_cn_a = jnp.where(h_placement_ok, cn_a & ~_H_DISABLE_N_A, cn_a)
    h_cn_b = jnp.where(h_placement_ok, cn_b & ~_H_DISABLE_N_B, cn_b)
    h_cn_c = jnp.where(h_placement_ok, cn_c & ~_H_DISABLE_N_C, cn_c)
    h_cs_a = jnp.where(h_placement_ok, cs_a & ~_H_DISABLE_S_A, cs_a)
    h_cs_b = jnp.where(h_placement_ok, cs_b & ~_H_DISABLE_S_B, cs_b)
    h_cs_c = jnp.where(h_placement_ok, cs_c & ~_H_DISABLE_S_C, cs_c)

    def h_bfs(h_cn_a, h_cn_b, h_cn_c, h_cs_a, h_cs_b, h_cs_c):
        return _has_path_both_bits(
            pawn0,
            pawn1,
            h_cn_a,
            h_cn_b,
            h_cn_c,
            h_cs_a,
            h_cs_b,
            h_cs_c,
            ce_a,
            ce_b,
            ce_c,
            cw_a,
            cw_b,
            cw_c,
        )

    h_paths = jax.vmap(h_bfs)(h_cn_a, h_cn_b, h_cn_c, h_cs_a, h_cs_b, h_cs_c)

    # Vertical walls: modify can_e and can_w bitmasks
    v_ce_a = jnp.where(v_placement_ok, ce_a & ~_V_DISABLE_E_A, ce_a)
    v_ce_b = jnp.where(v_placement_ok, ce_b & ~_V_DISABLE_E_B, ce_b)
    v_ce_c = jnp.where(v_placement_ok, ce_c & ~_V_DISABLE_E_C, ce_c)
    v_cw_a = jnp.where(v_placement_ok, cw_a & ~_V_DISABLE_W_A, cw_a)
    v_cw_b = jnp.where(v_placement_ok, cw_b & ~_V_DISABLE_W_B, cw_b)
    v_cw_c = jnp.where(v_placement_ok, cw_c & ~_V_DISABLE_W_C, cw_c)

    def v_bfs(v_ce_a, v_ce_b, v_ce_c, v_cw_a, v_cw_b, v_cw_c):
        return _has_path_both_bits(
            pawn0,
            pawn1,
            cn_a,
            cn_b,
            cn_c,
            cs_a,
            cs_b,
            cs_c,
            v_ce_a,
            v_ce_b,
            v_ce_c,
            v_cw_a,
            v_cw_b,
            v_cw_c,
        )

    v_paths = jax.vmap(v_bfs)(v_ce_a, v_ce_b, v_ce_c, v_cw_a, v_cw_b, v_cw_c)

    return jnp.concatenate([h_placement_ok & h_paths, v_placement_ok & v_paths])


def _move_legal_mask(
    state: GameState,
    can_n: Array = None,
    can_s: Array = None,
    can_e: Array = None,
    can_w: Array = None,
) -> Array:
    """Compute legality mask for 12 movement actions (4 cardinal + 4 jump + 4 diagonal).

    Optionally accepts pre-computed move masks to avoid redundant computation.
    """
    my_pos = state.pawn_pos[state.color]
    opp_pos = state.pawn_pos[1 - state.color]
    my_r, my_c = _pos_to_rc(my_pos)
    opp_r, opp_c = _pos_to_rc(opp_pos)

    if can_n is None:
        can_n, can_s, can_e, can_w = _precompute_move_masks(state.h_walls, state.v_walls)

    # Stack masks: can_dir[d][flat_pos] = can move from flat_pos in direction d
    # Direction order: 0=S(DR=-1), 1=N(DR=+1), 2=E(DC=+1), 3=W(DC=-1)
    can_dir = jnp.stack([can_s, can_n, can_e, can_w])  # [4, 81]

    def _not_blocked(r, c, d):
        """Check if movement from (r,c) in direction d is not blocked by wall."""
        flat = r * BOARD_SIZE + c
        return can_dir[d, flat]

    def check_cardinal(d):
        nr = my_r + DR[d]
        nc = my_c + DC[d]
        in_bounds = (nr >= 0) & (nr < BOARD_SIZE) & (nc >= 0) & (nc < BOARD_SIZE)
        not_blocked = _not_blocked(my_r, my_c, d)
        occupied = (nr == opp_r) & (nc == opp_c)
        return in_bounds & not_blocked & ~occupied

    def check_jump(d):
        adj_r = my_r + DR[d]
        adj_c = my_c + DC[d]
        opp_adjacent = (adj_r == opp_r) & (adj_c == opp_c)
        not_blocked_to_opp = _not_blocked(my_r, my_c, d)

        jump_r = opp_r + DR[d]
        jump_c = opp_c + DC[d]
        jump_in_bounds = (
            (jump_r >= 0) & (jump_r < BOARD_SIZE) & (jump_c >= 0) & (jump_c < BOARD_SIZE)
        )
        not_blocked_beyond = _not_blocked(opp_r, opp_c, d)

        return opp_adjacent & not_blocked_to_opp & jump_in_bounds & not_blocked_beyond

    def check_diagonal(d):
        first_dirs = jnp.array([[0, 2], [0, 3], [1, 2], [1, 3]], dtype=jnp.int32)

        def try_diagonal_via(approach_dir, slide_dir):
            adj_r = my_r + DR[approach_dir]
            adj_c = my_c + DC[approach_dir]
            opp_adjacent = (adj_r == opp_r) & (adj_c == opp_c)
            not_blocked_to_opp = _not_blocked(my_r, my_c, approach_dir)

            jump_r = opp_r + DR[approach_dir]
            jump_c = opp_c + DC[approach_dir]
            straight_blocked = (
                (jump_r < 0)
                | (jump_r >= BOARD_SIZE)
                | (jump_c < 0)
                | (jump_c >= BOARD_SIZE)
                | ~_not_blocked(opp_r, opp_c, approach_dir)
            )

            diag_r = opp_r + DR[slide_dir]
            diag_c = opp_c + DC[slide_dir]
            diag_in_bounds = (
                (diag_r >= 0) & (diag_r < BOARD_SIZE) & (diag_c >= 0) & (diag_c < BOARD_SIZE)
            )
            not_blocked_slide = _not_blocked(opp_r, opp_c, slide_dir)

            return (
                opp_adjacent
                & not_blocked_to_opp
                & straight_blocked
                & diag_in_bounds
                & not_blocked_slide
            )

        approach0 = first_dirs[d, 0]
        slide0 = first_dirs[d, 1]
        approach1 = first_dirs[d, 1]
        slide1 = first_dirs[d, 0]

        return try_diagonal_via(approach0, slide0) | try_diagonal_via(approach1, slide1)

    cardinal = jnp.array([check_cardinal(d) for d in range(4)])
    jumps = jnp.array([check_jump(d) for d in range(4)])
    diagonals = jnp.array([check_diagonal(d) for d in range(4)])

    return jnp.concatenate([cardinal, jumps, diagonals])


def _apply_move(state: GameState, action: Array) -> GameState:
    """Apply a movement action (0-11) to the game state."""
    my_pos = state.pawn_pos[state.color]
    opp_pos = state.pawn_pos[1 - state.color]
    my_r, my_c = _pos_to_rc(my_pos)
    opp_r, opp_c = _pos_to_rc(opp_pos)

    # Cardinal moves (0-3)
    card_r = my_r + DR[action]
    card_c = my_c + DC[action]

    # Jump moves (4-7): jump over opponent
    jump_dir = action - 4
    jump_r = opp_r + DR[jump_dir]
    jump_c = opp_c + DC[jump_dir]

    # Diagonal moves (8-11): destination is always my_pos + diagonal offset
    # (legality already validated by legal_action_mask)
    # SE(8), SW(9), NE(10), NW(11)
    diag_dr = jnp.array([-1, -1, 1, 1], dtype=jnp.int32)
    diag_dc = jnp.array([1, -1, 1, -1], dtype=jnp.int32)
    diag_idx = action - 8
    diag_r = my_r + diag_dr[diag_idx]
    diag_c = my_c + diag_dc[diag_idx]

    # Select target based on action type
    is_cardinal = action < 4
    is_jump = (action >= 4) & (action < 8)

    new_r = jnp.where(is_cardinal, card_r, jnp.where(is_jump, jump_r, diag_r))
    new_c = jnp.where(is_cardinal, card_c, jnp.where(is_jump, jump_c, diag_c))

    new_pos = _rc_to_pos(new_r, new_c)
    new_pawn_pos = state.pawn_pos.at[state.color].set(new_pos)

    # Check win: player 0 reaches row 8, player 1 reaches row 0
    goal_row = jnp.where(state.color == 0, BOARD_SIZE - 1, 0)
    won = new_r == goal_row
    new_winner = jnp.where(won, state.color, state.winner)

    return state._replace(
        pawn_pos=new_pawn_pos,
        winner=new_winner,
        color=1 - state.color,
    )


def _apply_wall(state: GameState, action: Array) -> GameState:
    """Apply a wall placement action (12-139)."""
    is_horizontal = action < 76
    wall_idx = jnp.where(is_horizontal, action - 12, action - 76)
    wall_r = wall_idx // WALL_SIZE
    wall_c = wall_idx % WALL_SIZE

    new_h_walls = jnp.where(
        is_horizontal, state.h_walls.at[wall_r, wall_c].set(True), state.h_walls
    )
    new_v_walls = jnp.where(
        ~is_horizontal, state.v_walls.at[wall_r, wall_c].set(True), state.v_walls
    )

    owner = state.color.astype(jnp.int8)
    new_h_owners = jnp.where(
        is_horizontal, state.h_wall_owners.at[wall_r, wall_c].set(owner), state.h_wall_owners
    )
    new_v_owners = jnp.where(
        ~is_horizontal, state.v_wall_owners.at[wall_r, wall_c].set(owner), state.v_wall_owners
    )

    new_walls_remaining = state.walls_remaining.at[state.color].add(-1)

    # Incrementally update cached move masks
    # Horizontal wall at (r,c) disables:
    #   can_n at cells r*9+c, r*9+c+1
    #   can_s at cells (r+1)*9+c, (r+1)*9+c+1
    h_cn = state.can_n.at[wall_r * 9 + wall_c].set(False).at[wall_r * 9 + wall_c + 1].set(False)
    h_cs = (
        state.can_s.at[(wall_r + 1) * 9 + wall_c]
        .set(False)
        .at[(wall_r + 1) * 9 + wall_c + 1]
        .set(False)
    )

    # Vertical wall at (r,c) disables:
    #   can_e at cells r*9+c, (r+1)*9+c
    #   can_w at cells r*9+c+1, (r+1)*9+c+1
    v_ce = state.can_e.at[wall_r * 9 + wall_c].set(False).at[(wall_r + 1) * 9 + wall_c].set(False)
    v_cw = (
        state.can_w.at[wall_r * 9 + wall_c + 1]
        .set(False)
        .at[(wall_r + 1) * 9 + wall_c + 1]
        .set(False)
    )

    new_can_n = jnp.where(is_horizontal, h_cn, state.can_n)
    new_can_s = jnp.where(is_horizontal, h_cs, state.can_s)
    new_can_e = jnp.where(~is_horizontal, v_ce, state.can_e)
    new_can_w = jnp.where(~is_horizontal, v_cw, state.can_w)

    return state._replace(
        h_walls=new_h_walls,
        v_walls=new_v_walls,
        h_wall_owners=new_h_owners,
        v_wall_owners=new_v_owners,
        walls_remaining=new_walls_remaining,
        color=1 - state.color,
        can_n=new_can_n,
        can_s=new_can_s,
        can_e=new_can_e,
        can_w=new_can_w,
    )


class Game:
    """Pure JAX Quoridor game logic."""

    def init(self) -> GameState:
        return GameState()

    def step(self, state: GameState, action: Array) -> GameState:
        is_move = action < 12
        move_state = _apply_move(state, jnp.clip(action, 0, 11))
        wall_state = _apply_wall(state, jnp.clip(action, 12, NUM_ACTIONS - 1))
        return jax.tree.map(lambda m, w: jnp.where(is_move, m, w), move_state, wall_state)

    def legal_action_mask(self, state: GameState) -> Array:
        move_mask = _move_legal_mask(state, state.can_n, state.can_s, state.can_e, state.can_w)
        wall_mask = _wall_legal_mask(state, state.can_n, state.can_s, state.can_e, state.can_w)
        return jnp.concatenate([move_mask, wall_mask])

    def is_terminal(self, state: GameState) -> Array:
        return state.winner >= 0

    def rewards(self, state: GameState) -> Array:
        return jax.lax.select(
            state.winner >= 0,
            jnp.float32([-1, -1]).at[state.winner].set(1),
            jnp.zeros(2, jnp.float32),
        )
