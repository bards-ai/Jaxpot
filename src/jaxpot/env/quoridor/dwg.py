"""SVG visualization for Quoridor board state, matching PGX style."""

from __future__ import annotations

import numpy as np

from .game import BOARD_SIZE, WALL_SIZE


def _make_quoridor_dwg(dwg, state, config):
    """PGX-compatible drawing function.

    Receives an svgwrite.Drawing, state, and config dict with
    GRID_SIZE, BOARD_WIDTH, BOARD_HEIGHT, COLOR_SET.
    Returns an svgwrite Group.
    """

    GRID_SIZE = config["GRID_SIZE"]
    color_set = config["COLOR_SET"]

    # Total board pixel size: 9 cells + 8 gaps
    gap = GRID_SIZE // 5  # gap between cells for wall slots
    cell = GRID_SIZE
    span = cell + gap  # center-to-center distance between cells

    board_px = BOARD_SIZE * cell + (BOARD_SIZE - 1) * gap
    board_g = dwg.g()

    # Background
    board_g.add(
        dwg.rect(
            (0, 0),
            (board_px, board_px),
            fill=color_set.background_color,
        )
    )

    # Grid lines (light)
    for i in range(BOARD_SIZE + 1):
        y = i * span - gap / 2 if i > 0 else 0
        if i == BOARD_SIZE:
            y = board_px
        # Horizontal
        board_g.add(
            dwg.line(
                start=(0, y),
                end=(board_px, y),
                stroke=color_set.grid_color,
                stroke_width="0.3px",
                stroke_opacity="0.3",
            )
        )
        # Vertical
        board_g.add(
            dwg.line(
                start=(y, 0),
                end=(y, board_px),
                stroke=color_set.grid_color,
                stroke_width="0.3px",
                stroke_opacity="0.3",
            )
        )

    # Board border
    board_g.add(
        dwg.rect(
            (0, 0),
            (board_px, board_px),
            fill="none",
            stroke=color_set.grid_color,
            stroke_width="2px",
        )
    )

    # Goal row indicators (thin lines at top and bottom edges)
    # Player 0 (Black) goal = row 9 (top of SVG, grid row 8)
    board_g.add(
        dwg.rect(
            (0, 0),
            (board_px, 3),
            fill=color_set.grid_color,
            fill_opacity="0.15",
        )
    )
    # Player 1 (White) goal = row 1 (bottom of SVG, grid row 0)
    board_g.add(
        dwg.rect(
            (0, board_px - 3),
            (board_px, 3),
            fill=color_set.grid_color,
            fill_opacity="0.15",
        )
    )

    # Extract state
    pawn_pos = np.array(state._x.pawn_pos)
    h_walls = np.array(state._x.h_walls)
    v_walls = np.array(state._x.v_walls)
    h_wall_owners = np.array(state._x.h_wall_owners)
    v_wall_owners = np.array(state._x.v_wall_owners)
    walls_remaining = np.array(state._x.walls_remaining)

    def _wall_style(owner: int) -> tuple[str, str]:
        """Return (fill, stroke) for a wall based on its owner."""
        if owner == 0:
            return color_set.p1_wall_color, color_set.p1_wall_outline
        elif owner == 1:
            return color_set.p2_wall_color, color_set.p2_wall_outline
        return color_set.grid_color, color_set.grid_color

    def cell_center(row, col):
        cx = col * span + cell / 2
        cy = (BOARD_SIZE - 1 - row) * span + cell / 2
        return cx, cy

    # Draw wall slots (subtle dotted grid in the gaps)
    for r in range(WALL_SIZE):
        for c in range(WALL_SIZE):
            # Center of wall intersection (y-flipped: row 0 at bottom)
            wx = (c + 1) * cell + c * gap + gap / 2
            wy = (BOARD_SIZE - 2 - r) * span + cell + gap / 2
            board_g.add(
                dwg.circle(
                    center=(wx, wy),
                    r=1,
                    fill=color_set.grid_color,
                    fill_opacity="0.15",
                )
            )

    # Draw horizontal walls
    for r in range(WALL_SIZE):
        for c in range(WALL_SIZE):
            if h_walls[r, c]:
                # Wall between rows r and r+1 (y-flipped: gap above row r in SVG)
                x0 = c * span
                y0 = (BOARD_SIZE - 2 - r) * span + cell
                w = 2 * cell + gap
                h = gap
                fill, stroke = _wall_style(int(h_wall_owners[r, c]))
                board_g.add(
                    dwg.rect(
                        (x0, y0),
                        (w, h),
                        fill=fill,
                        stroke=stroke,
                        stroke_width="1.5px",
                        rx=1,
                    )
                )

    # Draw vertical walls
    for r in range(WALL_SIZE):
        for c in range(WALL_SIZE):
            if v_walls[r, c]:
                x0 = (c + 1) * cell + c * gap
                y0 = (BOARD_SIZE - 2 - r) * span
                w = gap
                h = 2 * cell + gap
                fill, stroke = _wall_style(int(v_wall_owners[r, c]))
                board_g.add(
                    dwg.rect(
                        (x0, y0),
                        (w, h),
                        fill=fill,
                        stroke=stroke,
                        stroke_width="1.5px",
                        rx=1,
                    )
                )

    # Draw pawns
    # Player 0 (first player) = filled black, Player 1 = filled white with outline
    for player in range(2):
        pos = int(pawn_pos[player])
        pr, pc = pos // BOARD_SIZE, pos % BOARD_SIZE
        cx, cy = cell_center(pr, pc)
        r = cell * 0.35
        if player == 0:
            board_g.add(
                dwg.circle(
                    center=(cx, cy),
                    r=r,
                    fill=color_set.p1_color,
                    stroke=color_set.p1_outline,
                    stroke_width="1.5px",
                )
            )
        else:
            board_g.add(
                dwg.circle(
                    center=(cx, cy),
                    r=r,
                    fill=color_set.p2_color,
                    stroke=color_set.p2_outline,
                    stroke_width="1.5px",
                )
            )

    # Coordinate labels
    label_size = cell * 0.32
    label_opacity = "0.45"
    label_color = color_set.text_color
    col_letters = "abcdefghi"
    for c in range(BOARD_SIZE):
        cx, _ = cell_center(0, c)
        board_g.add(
            dwg.text(
                col_letters[c],
                insert=(cx, board_px + label_size * 1.1),
                fill=label_color,
                font_size=f"{label_size}px",
                font_family="sans-serif",
                text_anchor="middle",
                fill_opacity=label_opacity,
            )
        )
    for r in range(BOARD_SIZE):
        _, cy = cell_center(r, 0)
        board_g.add(
            dwg.text(
                str(r + 1),
                insert=(-label_size * 0.5, cy + label_size * 0.35),
                fill=label_color,
                font_size=f"{label_size}px",
                font_family="sans-serif",
                text_anchor="middle",
                fill_opacity=label_opacity,
            )
        )

    # Wall counts with pawn indicators (below column labels)
    walls_y = board_px + cell * 0.95
    wall_font = cell * 0.38
    dot_r = cell * 0.13
    dot_y = walls_y - wall_font * 0.3

    for player, px_x in ((0, cell * 0.6), (1, board_px - cell * 1.2)):
        p_color = color_set.p1_color if player == 0 else color_set.p2_color
        p_outline = color_set.p1_outline if player == 0 else color_set.p2_outline
        board_g.add(
            dwg.circle(
                center=(px_x, dot_y),
                r=dot_r,
                fill=p_color,
                stroke=p_outline,
                stroke_width="1px",
            )
        )
        board_g.add(
            dwg.text(
                f"{int(walls_remaining[player])}",
                insert=(px_x + dot_r + 3, walls_y),
                fill=label_color,
                font_size=f"{wall_font}px",
                font_family="sans-serif",
                fill_opacity="0.7",
            )
        )

    return board_g
