"""Pure JAX game logic for Phantom Tic-Tac-Toe.

Board is 3x3. Two players (X=player 0, O=player 1) take turns.
Each player can only see their own marks and cells revealed by failed moves.

Action space: 9 discrete actions (cells 0-8, row-major).

Cell encoding: 0=empty, 1=X (player 0), 2=O (player 1).

Two game variants:
  - classical: failed move reveals cell, player retries (same turn)
  - abrupt: failed move reveals cell, turn passes to opponent
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

NUM_CELLS = 9
NUM_ACTIONS = 9

# Win patterns as cell index triples
_WIN_PATTERNS = jnp.array([
    [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8],  # cols
    [0, 4, 8], [2, 4, 6],             # diagonals
], dtype=jnp.int32)


class GameState(NamedTuple):
    """Internal game state for Phantom Tic-Tac-Toe."""
    color: Array = jnp.int32(0)  # whose turn (0=X, 1=O)
    board: Array = jnp.zeros(9, dtype=jnp.int32)  # true board: 0=empty, 1=X, 2=O
    x_view: Array = jnp.zeros(9, dtype=jnp.int32)  # player 0's view
    o_view: Array = jnp.zeros(9, dtype=jnp.int32)  # player 1's view
    winner: Array = jnp.int32(-1)  # -1=ongoing, 0=X wins, 1=O wins, 2=draw
    move_succeeded: Array = jnp.bool_(True)  # did last move place a mark?


def _check_winner(board: Array, player_mark: int) -> Array:
    """Check if player_mark (1 or 2) has three in a row."""
    cells = board[_WIN_PATTERNS]  # (8, 3)
    return jnp.any(jnp.all(cells == player_mark, axis=1))


def _board_full(board: Array) -> Array:
    """Check if all cells are occupied."""
    return jnp.all(board != 0)


def _get_view(state: GameState) -> Array:
    """Get the current player's view."""
    return jax.lax.select(state.color == 0, state.x_view, state.o_view)


def _legal_action_mask(state: GameState) -> Array:
    """Legal actions = cells that appear empty in the current player's view."""
    view = _get_view(state)
    return view == 0


def _step(state: GameState, action: Array, abrupt: bool = False) -> GameState:
    """Execute one action attempt.

    Args:
        state: Current game state.
        action: Cell index (0-8) to attempt placement.
        abrupt: If True, failed moves lose the turn.

    Returns:
        New game state after the attempt.
    """
    mark = state.color + 1  # 1 for X, 2 for O
    cell_empty = state.board[action] == 0

    # --- Successful move: place mark on true board ---
    new_board = jnp.where(cell_empty, state.board.at[action].set(mark), state.board)

    # Update the acting player's view with their own mark
    new_x_view = jnp.where(
        (state.color == 0) & cell_empty,
        state.x_view.at[action].set(mark),
        state.x_view,
    )
    new_o_view = jnp.where(
        (state.color == 1) & cell_empty,
        state.o_view.at[action].set(mark),
        state.o_view,
    )

    # --- Failed move: reveal the true cell contents to the acting player ---
    true_cell = state.board[action]
    new_x_view = jnp.where(
        (state.color == 0) & ~cell_empty,
        new_x_view.at[action].set(true_cell),
        new_x_view,
    )
    new_o_view = jnp.where(
        (state.color == 1) & ~cell_empty,
        new_o_view.at[action].set(true_cell),
        new_o_view,
    )

    # Check terminal conditions (only after successful moves)
    x_wins = _check_winner(new_board, 1)
    o_wins = _check_winner(new_board, 2)
    draw = _board_full(new_board) & ~x_wins & ~o_wins

    new_winner = jnp.where(
        cell_empty & x_wins, jnp.int32(0),
        jnp.where(
            cell_empty & o_wins, jnp.int32(1),
            jnp.where(cell_empty & draw, jnp.int32(2), state.winner),
        ),
    )

    # Turn switching:
    # - Successful move: always switch
    # - Failed move classical: stay (same player retries)
    # - Failed move abrupt: switch
    if abrupt:
        # Always switch turns
        new_color = 1 - state.color
    else:
        # Classical: only switch on success
        new_color = jnp.where(cell_empty, 1 - state.color, state.color)

    return GameState(
        color=new_color,
        board=new_board,
        x_view=new_x_view,
        o_view=new_o_view,
        winner=new_winner,
        move_succeeded=cell_empty,
    )


class Game:
    """Pure JAX Phantom Tic-Tac-Toe game logic."""

    def __init__(self, abrupt: bool = False):
        self._abrupt = abrupt

    def init(self) -> GameState:
        return GameState()

    def step(self, state: GameState, action: Array) -> GameState:
        return _step(state, action, abrupt=self._abrupt)

    def legal_action_mask(self, state: GameState) -> Array:
        return _legal_action_mask(state)

    def is_terminal(self, state: GameState) -> Array:
        return state.winner >= 0

    def rewards(self, state: GameState) -> Array:
        """Returns (2,) rewards. Zero-sum: winner gets +1, loser gets -1, draw is 0."""
        return jax.lax.switch(
            jnp.clip(state.winner, 0, 2),
            [
                lambda: jnp.float32([1.0, -1.0]),   # X wins
                lambda: jnp.float32([-1.0, 1.0]),   # O wins
                lambda: jnp.float32([0.0, 0.0]),     # draw
            ],
        )
