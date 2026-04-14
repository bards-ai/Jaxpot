"""SVG visualization for Dark Hex, matching PGX style.

Renders two views side by side: player 0 (Black) view and player 1 (White) view,
plus the true board in the center.

Hex cells are rendered as pointy-top hexagons on a staggered grid.
"""

from __future__ import annotations

import math

import numpy as np


def _pointy_hex_center(row, col, size):
    """Compute pixel center for pointy-top hexagons (standard Hex board layout).

    Each row is staggered to the right by half a hex width.
    """
    w = size * math.sqrt(3)
    cx = w / 2 + col * w + row * w / 2
    cy = size * 1.5 * row + size
    return cx, cy


def _pointy_hex_points(cx, cy, size):
    """Compute 6 vertices of a pointy-top hexagon."""
    points = []
    for i in range(6):
        angle_deg = 60 * i - 30
        angle_rad = math.radians(angle_deg)
        px = cx + size * math.cos(angle_rad)
        py = cy + size * math.sin(angle_rad)
        points.append((px, py))
    return points


def _board_pixel_size(num_rows, num_cols, hex_size):
    """Compute pixel dimensions of a hex board."""
    w = hex_size * math.sqrt(3)
    # Width: num_cols hexes + half-hex stagger per row
    board_w = w * num_cols + w * (num_rows - 1) / 2 + w * 0.1
    # Height: stacked rows at 1.5 * size spacing + top/bottom padding
    board_h = hex_size * 1.5 * (num_rows - 1) + hex_size * 2 + hex_size * 0.1
    return board_w, board_h


def _draw_hex_board(
    dwg, g, board, num_rows, num_cols, hex_size, cell_color, grid_color, black_color, white_color
):
    """Draw a hex board with stones.

    Args:
        board: (num_cells,) array with 0=empty, 1=Black, 2=White.
    """
    for r in range(num_rows):
        for c in range(num_cols):
            idx = r * num_cols + c
            cx, cy = _pointy_hex_center(r, c, hex_size)
            pts = _pointy_hex_points(cx, cy, hex_size * 0.92)

            # Draw hexagon cell
            g.add(
                dwg.polygon(
                    points=pts,
                    fill=cell_color,
                    stroke=grid_color,
                    stroke_width=hex_size * 0.05,
                )
            )

            # Draw stone
            mark = int(board[idx])
            if mark == 1:  # Black
                g.add(
                    dwg.circle(
                        center=(cx, cy),
                        r=hex_size * 0.38,
                        fill=black_color,
                        stroke=grid_color,
                        stroke_width=hex_size * 0.03,
                    )
                )
            elif mark == 2:  # White
                g.add(
                    dwg.circle(
                        center=(cx, cy),
                        r=hex_size * 0.38,
                        fill=white_color,
                        stroke=grid_color,
                        stroke_width=hex_size * 0.03,
                    )
                )


def _hex_pts(row, col, hex_size, scale=0.92):
    """Get hex vertices for cell (row, col).

    Pointy-top hex vertices in SVG coords (y-down):
      0: top-right    1: right (bottom-right)
      2: bottom       3: bottom-left
      4: top-left     5: top
    """
    cx, cy = _pointy_hex_center(row, col, hex_size)
    return _pointy_hex_points(cx, cy, hex_size * scale)


def _draw_border_indicators(dwg, g, num_rows, num_cols, hex_size, grid_color):
    """Draw hex-edge zig-zag borders to indicate player goals.

    Black zig-zag on North/South edges (Black connects top to bottom).
    White segments (with black outline) on West/East edges (White connects left to right).
    """
    sw = hex_size * 0.1
    border_scale = 1.0
    marker_offset = hex_size * 0.16

    def _line_intersection(p0, p1, p2, p3):
        x1, y1 = p0
        x2, y2 = p1
        x3, y3 = p2
        x4, y4 = p3
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-9:
            return p1
        det1 = x1 * y2 - y1 * x2
        det2 = x3 * y4 - y3 * x4
        px = (det1 * (x3 - x4) - (x1 - x2) * det2) / denom
        py = (det1 * (y3 - y4) - (y1 - y2) * det2) / denom
        return px, py

    def _offset_polyline(points, offset):
        if len(points) < 2:
            return points

        offset_segments = []
        for start, end in zip(points[:-1], points[1:]):
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = math.hypot(dx, dy)
            nx = -dy / length
            ny = dx / length
            offset_segments.append(
                (
                    (start[0] + offset * nx, start[1] + offset * ny),
                    (end[0] + offset * nx, end[1] + offset * ny),
                )
            )

        offset_points = [offset_segments[0][0]]
        for prev_seg, next_seg in zip(offset_segments[:-1], offset_segments[1:]):
            offset_points.append(
                _line_intersection(prev_seg[0], prev_seg[1], next_seg[0], next_seg[1])
            )
        offset_points.append(offset_segments[-1][1])
        return offset_points

    def _shift(points, dx=0.0, dy=0.0):
        return [(x + dx, y + dy) for x, y in points]

    # --- North border (Black): zig-zag along top edges of row 0 ---
    north_pts = [_hex_pts(0, 0, hex_size, scale=border_scale)[4]]
    for c in range(num_cols):
        pts = _hex_pts(0, c, hex_size, scale=border_scale)
        north_pts.extend([pts[5], pts[0]])
    g.add(
        dwg.polyline(
            points=_shift(north_pts, dy=-marker_offset),
            fill="none",
            stroke="black",
            stroke_width=sw,
            stroke_linecap="round",
            stroke_linejoin="round",
        )
    )

    # --- South border (Black): zig-zag along bottom edges of last row ---
    south_pts = [_hex_pts(num_rows - 1, 0, hex_size, scale=border_scale)[3]]
    for c in range(num_cols):
        pts = _hex_pts(num_rows - 1, c, hex_size, scale=border_scale)
        south_pts.extend([pts[2], pts[1]])
    g.add(
        dwg.polyline(
            points=_shift(south_pts, dy=marker_offset),
            fill="none",
            stroke="black",
            stroke_width=sw,
            stroke_linecap="round",
            stroke_linejoin="round",
        )
    )

    # --- West border (White): zig-zag along left outline ---
    west_pts = [_hex_pts(0, 0, hex_size, scale=border_scale)[4]]
    for r in range(num_rows):
        pts = _hex_pts(r, 0, hex_size, scale=border_scale)
        west_pts.extend([pts[3]])
        if r < num_rows - 1:
            next_pts = _hex_pts(r + 1, 0, hex_size, scale=border_scale)
            west_pts.extend([next_pts[4]])
    west_pts = _offset_polyline(west_pts, marker_offset)
    g.add(
        dwg.polyline(
            points=west_pts,
            fill="none",
            stroke="black",
            stroke_width=sw + hex_size * 0.06,
            stroke_linecap="round",
            stroke_linejoin="round",
        )
    )
    g.add(
        dwg.polyline(
            points=west_pts,
            fill="none",
            stroke="white",
            stroke_width=sw,
            stroke_linecap="round",
            stroke_linejoin="round",
        )
    )

    # --- East border (White): zig-zag along right outline ---
    east_pts = [_hex_pts(0, num_cols - 1, hex_size, scale=border_scale)[0]]
    for r in range(num_rows):
        pts = _hex_pts(r, num_cols - 1, hex_size, scale=border_scale)
        east_pts.extend([pts[1]])
        if r < num_rows - 1:
            next_pts = _hex_pts(r + 1, num_cols - 1, hex_size, scale=border_scale)
            east_pts.extend([next_pts[0]])
    east_pts = _offset_polyline(east_pts, -marker_offset)
    g.add(
        dwg.polyline(
            points=east_pts,
            fill="none",
            stroke="black",
            stroke_width=sw + hex_size * 0.06,
            stroke_linecap="round",
            stroke_linejoin="round",
        )
    )
    g.add(
        dwg.polyline(
            points=east_pts,
            fill="none",
            stroke="white",
            stroke_width=sw,
            stroke_linecap="round",
            stroke_linejoin="round",
        )
    )


def _make_dark_hex_dwg(dwg, state, config):
    """Render a single Dark Hex state as three boards side by side.

    Layout: [Black's view] [True board] [White's view]
    """
    GRID_SIZE = config["GRID_SIZE"]
    color_set = config["COLOR_SET"]
    num_rows = config.get("NUM_ROWS", 3)
    num_cols = config.get("NUM_COLS", 3)

    board = np.array(state._x.board)
    black_view = np.array(state._x.black_view)
    white_view = np.array(state._x.white_view)

    hex_size = GRID_SIZE * 0.6
    board_w, board_h = _board_pixel_size(num_rows, num_cols, hex_size)

    spacing = GRID_SIZE * 0.6
    label_size = GRID_SIZE * 0.32
    pad = hex_size * 0.3  # padding around board for edge markers

    black_stone = "black"
    white_stone = "white"
    cell_color = color_set.background_color
    grid_color = color_set.grid_color
    label_color = color_set.text_color

    root_g = dwg.g()

    def _draw_panel(parent_g, board_data, label, offset_x, show_borders=False):
        pg = dwg.g()
        pg.translate(offset_x, 0)

        bg = dwg.g()
        bg.translate(pad, pad)
        _draw_hex_board(
            dwg,
            bg,
            board_data,
            num_rows,
            num_cols,
            hex_size,
            cell_color,
            grid_color,
            black_stone,
            white_stone,
        )
        if show_borders:
            _draw_border_indicators(dwg, bg, num_rows, num_cols, hex_size, grid_color)
        pg.add(bg)

        # Label below
        panel_w = board_w + 2 * pad
        panel_h = board_h + 2 * pad
        pg.add(
            dwg.text(
                label,
                insert=(panel_w / 2, panel_h + label_size * 1.2),
                fill=label_color,
                font_size=f"{label_size}px",
                font_family="sans-serif",
                text_anchor="middle",
                fill_opacity="0.6",
            )
        )
        parent_g.add(pg)
        return panel_w

    panel_w = board_w + 2 * pad

    _draw_panel(root_g, black_view, "Black view", 0)
    _draw_panel(root_g, board, "True board", panel_w + spacing, show_borders=True)
    _draw_panel(root_g, white_view, "White view", 2 * (panel_w + spacing))

    # Turn indicator under center board
    turn = "Black" if int(state._x.color) == 0 else "White"
    center_x = panel_w + spacing + panel_w / 2
    center_y = board_h + 2 * pad + label_size * 2.4
    root_g.add(
        dwg.text(
            f"{turn} to move",
            insert=(center_x, center_y),
            fill=label_color,
            font_size=f"{label_size * 0.85}px",
            font_family="sans-serif",
            text_anchor="middle",
            fill_opacity="0.4",
        )
    )

    return root_g
