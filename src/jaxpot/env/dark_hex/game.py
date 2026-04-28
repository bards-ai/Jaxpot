"""Pure JAX game logic for Dark Hex.

Dark Hex is an imperfect-information variant of Hex. Two players alternate
placing stones on a hex grid, but neither player can see the opponent's stones.

Player 0 (Black) connects North (row 0) to South (row num_rows-1).
Player 1 (White) connects West (col 0) to East (col num_cols-1).

Action space: num_rows * num_cols discrete actions (cell indices, row-major).

Cell encoding: 0=empty, 1=Black (player 0), 2=White (player 1).

Hex grid neighbors for cell at (row, col):
  North:     (row-1, col)
  NorthEast: (row-1, col+1)
  East:      (row,   col+1)
  South:     (row+1, col)
  SouthWest: (row+1, col-1)
  West:      (row,   col-1)

Two game variants:
  - classical: failed move reveals cell, player retries (same turn)
  - abrupt: failed move reveals cell, turn passes to opponent
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


class GameState(NamedTuple):
    """Internal game state for Dark Hex."""

    color: Array  # whose turn (0=Black, 1=White)
    board: Array  # true board: (num_cells,) 0=empty, 1=Black, 2=White
    black_view: Array  # player 0's view: (num_cells,)
    white_view: Array  # player 1's view: (num_cells,)
    winner: Array  # -1=ongoing, 0=Black wins, 1=White wins
    move_succeeded: Array  # did last move place a stone?


def _build_neighbor_table(num_rows: int, num_cols: int) -> np.ndarray:
    """Build hex neighbor lookup table.

    Returns (num_cells, 6) array where sentinel value num_cells means no neighbor.
    """
    num_cells = num_rows * num_cols
    sentinel = num_cells
    neighbors = np.full((num_cells, 6), sentinel, dtype=np.int32)

    for r in range(num_rows):
        for c in range(num_cols):
            idx = r * num_cols + c
            # 6 hex directions: N, NE, E, S, SW, W
            deltas = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]
            for d, (dr, dc) in enumerate(deltas):
                nr, nc = r + dr, c + dc
                if 0 <= nr < num_rows and 0 <= nc < num_cols:
                    neighbors[idx, d] = nr * num_cols + nc

    return neighbors


def _build_edge_masks(num_rows: int, num_cols: int):
    """Build edge masks for win detection.

    Returns:
        north_edge: (num_cells,) bool - row 0 cells (Black's start edge)
        south_edge: (num_cells,) bool - last row cells (Black's goal edge)
        west_edge:  (num_cells,) bool - col 0 cells (White's start edge)
        east_edge:  (num_cells,) bool - last col cells (White's goal edge)
    """
    num_cells = num_rows * num_cols
    north = np.zeros(num_cells, dtype=bool)
    south = np.zeros(num_cells, dtype=bool)
    west = np.zeros(num_cells, dtype=bool)
    east = np.zeros(num_cells, dtype=bool)

    for r in range(num_rows):
        for c in range(num_cols):
            idx = r * num_cols + c
            if r == 0:
                north[idx] = True
            if r == num_rows - 1:
                south[idx] = True
            if c == 0:
                west[idx] = True
            if c == num_cols - 1:
                east[idx] = True

    return north, south, west, east


def _check_connected(
    board: Array,
    mark: int,
    start_edge: Array,
    goal_edge: Array,
    neighbors: Array,
    num_iters: int,
) -> Array:
    """Check if player `mark` connects start_edge to goal_edge via flood fill.

    Uses iterative expansion: each iteration propagates reachability one hop
    through same-color stones. After num_iters iterations, checks if any
    goal_edge cell is reached.

    Args:
        board: (num_cells,) cell values.
        mark: Player mark to check (1=Black, 2=White).
        start_edge: (num_cells,) bool mask of starting edge cells.
        goal_edge: (num_cells,) bool mask of goal edge cells.
        neighbors: (num_cells, 6) neighbor indices with sentinel=num_cells.
        num_iters: Number of flood fill iterations (= num_cells).

    Returns:
        Scalar bool: whether start and goal edges are connected.
    """
    player_cells = board == mark  # (num_cells,)
    reached = start_edge & player_cells  # (num_cells,)

    # Pad with False sentinel for out-of-bounds neighbor lookups
    def body(_, reached):
        padded = jnp.concatenate([reached, jnp.array([False])])
        neighbor_reached = padded[neighbors]  # (num_cells, 6)
        any_neighbor = jnp.any(neighbor_reached, axis=1)  # (num_cells,)
        return reached | (player_cells & any_neighbor)

    reached = jax.lax.fori_loop(0, num_iters, body, reached)
    return jnp.any(reached & goal_edge)


class Game:
    """Pure JAX Dark Hex game logic.

    Args:
        num_rows: Board height. Default 3.
        num_cols: Board width. Default 3.
        abrupt: If True, failed moves lose the turn. Default False (classical).
    """

    def __init__(self, num_rows: int = 3, num_cols: int = 3, abrupt: bool = False):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_cells = num_rows * num_cols
        self.num_actions = self.num_cells
        self._abrupt = abrupt

        # Precompute static tables
        self._neighbors = jnp.array(_build_neighbor_table(num_rows, num_cols))
        north, south, west, east = _build_edge_masks(num_rows, num_cols)
        self._north_edge = jnp.array(north)
        self._south_edge = jnp.array(south)
        self._west_edge = jnp.array(west)
        self._east_edge = jnp.array(east)

    def init(self) -> GameState:
        return GameState(
            color=jnp.int32(0),
            board=jnp.zeros(self.num_cells, dtype=jnp.int32),
            black_view=jnp.zeros(self.num_cells, dtype=jnp.int32),
            white_view=jnp.zeros(self.num_cells, dtype=jnp.int32),
            winner=jnp.int32(-1),
            move_succeeded=jnp.bool_(True),
        )

    def step(self, state: GameState, action: Array) -> GameState:
        mark = state.color + 1  # 1 for Black, 2 for White
        cell_empty = state.board[action] == 0

        # --- Successful move: place stone ---
        new_board = jnp.where(cell_empty, state.board.at[action].set(mark), state.board)

        # Update acting player's view with their own stone
        new_black_view = jnp.where(
            (state.color == 0) & cell_empty,
            state.black_view.at[action].set(mark),
            state.black_view,
        )
        new_white_view = jnp.where(
            (state.color == 1) & cell_empty,
            state.white_view.at[action].set(mark),
            state.white_view,
        )

        # --- Failed move: reveal the true cell contents ---
        true_cell = state.board[action]
        new_black_view = jnp.where(
            (state.color == 0) & ~cell_empty,
            new_black_view.at[action].set(true_cell),
            new_black_view,
        )
        new_white_view = jnp.where(
            (state.color == 1) & ~cell_empty,
            new_white_view.at[action].set(true_cell),
            new_white_view,
        )

        # Check win conditions (only after successful moves)
        black_wins = cell_empty & _check_connected(
            new_board,
            1,
            self._north_edge,
            self._south_edge,
            self._neighbors,
            self.num_cells,
        )
        white_wins = cell_empty & _check_connected(
            new_board,
            2,
            self._west_edge,
            self._east_edge,
            self._neighbors,
            self.num_cells,
        )

        new_winner = jnp.where(
            black_wins,
            jnp.int32(0),
            jnp.where(white_wins, jnp.int32(1), state.winner),
        )

        # Turn switching
        if self._abrupt:
            new_color = 1 - state.color
        else:
            new_color = jnp.where(cell_empty, 1 - state.color, state.color)

        return GameState(
            color=new_color,
            board=new_board,
            black_view=new_black_view,
            white_view=new_white_view,
            winner=new_winner,
            move_succeeded=cell_empty,
        )

    def legal_action_mask(self, state: GameState) -> Array:
        """Legal actions = cells that appear empty in the current player's view."""
        view = jax.lax.select(state.color == 0, state.black_view, state.white_view)
        return view == 0

    def is_terminal(self, state: GameState) -> Array:
        return state.winner >= 0

    def rewards(self, state: GameState) -> Array:
        """Returns (2,) rewards. Zero-sum: winner gets +1, loser gets -1.

        Note: Hex cannot end in a draw.
        """
        return jax.lax.switch(
            jnp.clip(state.winner, 0, 1),
            [
                lambda: jnp.float32([1.0, -1.0]),  # Black wins
                lambda: jnp.float32([-1.0, 1.0]),  # White wins
            ],
        )
