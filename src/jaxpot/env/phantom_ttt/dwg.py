"""SVG visualization for Phantom Tic-Tac-Toe, matching PGX style.

Renders two views side by side: player 0 (X) view and player 1 (O) view,
plus the true board in the center. Hidden cells are shown with a fog overlay.
"""

from __future__ import annotations

import numpy as np


def _draw_x(dwg, g, cx, cy, size, color, stroke_width):
    """Draw an X mark centered at (cx, cy)."""
    d = size * 0.35
    g.add(
        dwg.line(
            start=(cx - d, cy - d),
            end=(cx + d, cy + d),
            stroke=color,
            stroke_width=stroke_width,
            stroke_linecap="round",
        )
    )
    g.add(
        dwg.line(
            start=(cx - d, cy + d),
            end=(cx + d, cy - d),
            stroke=color,
            stroke_width=stroke_width,
            stroke_linecap="round",
        )
    )


def _draw_o(dwg, g, cx, cy, size, color, stroke_width):
    """Draw an O mark centered at (cx, cy)."""
    g.add(
        dwg.circle(
            center=(cx, cy),
            r=size * 0.35,
            stroke=color,
            stroke_width=stroke_width,
            fill="none",
        )
    )


def _draw_grid(dwg, g, grid_size, color):
    """Draw the 3x3 grid lines."""
    total = grid_size * 3
    for i in range(1, 3):
        # Horizontal
        g.add(
            dwg.line(
                start=(0, grid_size * i),
                end=(total, grid_size * i),
                stroke=color,
                stroke_width=grid_size * 0.04,
            )
        )
        # Vertical
        g.add(
            dwg.line(
                start=(grid_size * i, 0),
                end=(grid_size * i, total),
                stroke=color,
                stroke_width=grid_size * 0.04,
            )
        )


def _draw_board(dwg, g, board, grid_size, grid_color, fog_mask=None):
    """Draw marks on a 3x3 board.

    Args:
        board: (9,) array with 0=empty, 1=X, 2=O.
        fog_mask: Optional (9,) bool array. True = cell is hidden (fog).
    """
    sw = grid_size * 0.06

    for i in range(9):
        x = i % 3
        y = i // 3
        cx = (x + 0.5) * grid_size
        cy = (y + 0.5) * grid_size

        # Fog overlay for hidden cells
        if fog_mask is not None and fog_mask[i]:
            g.add(
                dwg.rect(
                    (x * grid_size, y * grid_size),
                    (grid_size, grid_size),
                    fill=grid_color,
                    fill_opacity="0.08",
                )
            )

        mark = int(board[i])
        if mark == 1:  # X
            _draw_x(dwg, g, cx, cy, grid_size, grid_color, sw)
        elif mark == 2:  # O
            _draw_o(dwg, g, cx, cy, grid_size, grid_color, sw)


def _make_phantom_ttt_dwg(dwg, state, config):
    """Render a single Phantom TTT state as three boards side by side.

    Layout: [X's view] [True board] [O's view]
    """
    GRID_SIZE = config["GRID_SIZE"]
    color_set = config["COLOR_SET"]

    board = np.array(state._x.board)
    x_view = np.array(state._x.x_view)
    o_view = np.array(state._x.o_view)

    board_px = GRID_SIZE * 3
    spacing = GRID_SIZE * 0.8
    label_size = GRID_SIZE * 0.32
    label_color = color_set.text_color

    root_g = dwg.g()

    # --- Player X's view (left) ---
    x_g = dwg.g()
    # Background
    x_g.add(
        dwg.rect(
            (0, 0),
            (board_px, board_px),
            fill=color_set.background_color,
            stroke=color_set.grid_color,
            stroke_width=GRID_SIZE * 0.04,
        )
    )
    _draw_grid(dwg, x_g, GRID_SIZE, color_set.grid_color)
    _draw_board(dwg, x_g, x_view, GRID_SIZE, color_set.grid_color, fog_mask=None)
    # Label
    x_g.add(
        dwg.text(
            "X view",
            insert=(board_px / 2, board_px + label_size * 1.4),
            fill=label_color,
            font_size=f"{label_size}px",
            font_family="sans-serif",
            text_anchor="middle",
            fill_opacity="0.6",
        )
    )
    root_g.add(x_g)

    # --- True board (center) ---
    true_g = dwg.g()
    true_g.translate(board_px + spacing, 0)
    # Background
    true_g.add(
        dwg.rect(
            (0, 0),
            (board_px, board_px),
            fill=color_set.background_color,
            stroke=color_set.grid_color,
            stroke_width=GRID_SIZE * 0.04,
        )
    )
    _draw_grid(dwg, true_g, GRID_SIZE, color_set.grid_color)
    _draw_board(dwg, true_g, board, GRID_SIZE, color_set.grid_color)
    true_g.add(
        dwg.text(
            "True board",
            insert=(board_px / 2, board_px + label_size * 1.4),
            fill=label_color,
            font_size=f"{label_size}px",
            font_family="sans-serif",
            text_anchor="middle",
            fill_opacity="0.6",
        )
    )

    # Current player indicator
    turn_mark = "X" if int(state._x.color) == 0 else "O"
    true_g.add(
        dwg.text(
            f"{turn_mark} to move",
            insert=(board_px / 2, board_px + label_size * 2.6),
            fill=label_color,
            font_size=f"{label_size * 0.85}px",
            font_family="sans-serif",
            text_anchor="middle",
            fill_opacity="0.4",
        )
    )
    root_g.add(true_g)

    # --- Player O's view (right) ---
    o_g = dwg.g()
    o_g.translate(2 * (board_px + spacing), 0)
    # Background
    o_g.add(
        dwg.rect(
            (0, 0),
            (board_px, board_px),
            fill=color_set.background_color,
            stroke=color_set.grid_color,
            stroke_width=GRID_SIZE * 0.04,
        )
    )
    _draw_grid(dwg, o_g, GRID_SIZE, color_set.grid_color)
    _draw_board(dwg, o_g, o_view, GRID_SIZE, color_set.grid_color, fog_mask=None)
    o_g.add(
        dwg.text(
            "O view",
            insert=(board_px / 2, board_px + label_size * 1.4),
            fill=label_color,
            font_size=f"{label_size}px",
            font_family="sans-serif",
            text_anchor="middle",
            fill_opacity="0.6",
        )
    )
    root_g.add(o_g)

    return root_g
