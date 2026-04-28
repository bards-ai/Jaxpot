"""SVG visualization for Liar's Dice game state, matching PGX style."""

from __future__ import annotations

import numpy as np

# Pip (dot) positions for each die face value, in a 3x3 grid.
# Coordinates are (col, row) with origin top-left, in units of [0, 1].
_PIP_POSITIONS: dict[int, list[tuple[float, float]]] = {
    1: [(0.5, 0.5)],
    2: [(0.25, 0.25), (0.75, 0.75)],
    3: [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)],
    4: [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)],
    5: [(0.25, 0.25), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.75, 0.75)],
    6: [
        (0.25, 0.25), (0.75, 0.25),
        (0.25, 0.5), (0.75, 0.5),
        (0.25, 0.75), (0.75, 0.75),
    ],
}


def _draw_die(dwg, group, x, y, size, value, pip_color="black", bg="white",
              stroke="black", hidden=False):
    """Draw a single die at (x, y) with given size."""
    # Die body
    group.add(
        dwg.rect(
            (x, y), (size, size),
            fill=bg,
            stroke=stroke,
            stroke_width="1.5px",
            rx=size * 0.12,
            ry=size * 0.12,
        )
    )

    if hidden:
        # Draw a question mark for hidden dice
        group.add(
            dwg.text(
                "?",
                insert=(x + size * 0.5, y + size * 0.62),
                fill=stroke,
                font_size=f"{size * 0.55}px",
                font_family="sans-serif",
                font_weight="bold",
                text_anchor="middle",
                fill_opacity="0.3",
            )
        )
        return

    value = int(value)
    if value < 1 or value > 6:
        return

    pip_r = size * 0.09
    for px, py in _PIP_POSITIONS[value]:
        group.add(
            dwg.circle(
                center=(x + px * size, y + py * size),
                r=pip_r,
                fill=pip_color,
            )
        )


def _make_liars_dice_dwg(dwg, state, config):
    """PGX-compatible drawing function.

    Renders a god-view of the game: both players' dice, current bid,
    and game status.

    Parameters
    ----------
    dwg : svgwrite.Drawing
    state : State
    config : dict with GRID_SIZE, COLOR_SET

    Returns
    -------
    svgwrite.Group
    """
    from .game import DICE_SIDES, _bid_face, _bid_quantity, _liar_action

    GRID_SIZE = config["GRID_SIZE"]
    color_set = config["COLOR_SET"]

    dice = np.array(state._x.dice)
    num_dice = dice.shape[1]
    current_bid = int(np.array(state._x.current_bid))
    bidder = int(np.array(state._x.bidder))
    winner = int(np.array(state._x.winner))
    color = int(np.array(state._x.color))

    die_size = GRID_SIZE
    die_gap = GRID_SIZE * 0.25
    margin = GRID_SIZE * 0.5

    # Layout dimensions
    dice_row_w = num_dice * die_size + (num_dice - 1) * die_gap
    total_w = dice_row_w + 2 * margin
    label_h = GRID_SIZE * 0.55
    bid_h = GRID_SIZE * 0.8

    board_g = dwg.g()

    # Background
    total_h = (
        margin  # top margin
        + label_h  # "Player 0" label
        + die_size  # player 0 dice
        + GRID_SIZE * 0.4  # spacing
        + bid_h  # bid info
        + GRID_SIZE * 0.4  # spacing
        + die_size  # player 1 dice
        + label_h  # "Player 1" label
        + margin  # bottom margin
    )

    board_g.add(
        dwg.rect(
            (0, 0), (total_w, total_h),
            fill=color_set.background_color,
        )
    )
    board_g.add(
        dwg.rect(
            (0, 0), (total_w, total_h),
            fill="none",
            stroke=color_set.grid_color,
            stroke_width="1.5px",
        )
    )

    font_family = "ui-monospace, SFMono-Regular, Menlo, Monaco, monospace"
    text_color = color_set.text_color

    # --- Player 0 ---
    y_cursor = margin

    # Label
    p0_label = "Player 0"
    if winner == 0:
        p0_label += "  (winner)"
    elif color == 0 and winner < 0:
        p0_label += "  ←"
    board_g.add(
        dwg.text(
            p0_label,
            insert=(margin, y_cursor + label_h * 0.75),
            fill=text_color,
            font_size=f"{label_h * 0.7}px",
            font_family=font_family,
            fill_opacity="0.6",
        )
    )
    y_cursor += label_h

    # Player 0 dice
    for i in range(num_dice):
        dx = margin + i * (die_size + die_gap)
        _draw_die(dwg, board_g, dx, y_cursor, die_size, dice[0, i],
                  pip_color=color_set.p1_color, stroke=color_set.grid_color)
    y_cursor += die_size + GRID_SIZE * 0.4

    # --- Bid info ---
    if current_bid >= 0:
        quantity = int((current_bid // DICE_SIDES) + 1)
        face = int((current_bid % DICE_SIDES) + 1)
        bid_text = f"Bid: {quantity}x"
        bidder_text = f"by P{bidder}"
    else:
        bid_text = "No bids yet"
        quantity = 0
        face = 0
        bidder_text = ""

    # Separator line
    board_g.add(
        dwg.line(
            start=(margin, y_cursor),
            end=(total_w - margin, y_cursor),
            stroke=color_set.grid_color,
            stroke_width="0.5px",
            stroke_opacity="0.3",
        )
    )

    bid_font = bid_h * 0.45
    bid_text_y = y_cursor + bid_h * 0.6

    board_g.add(
        dwg.text(
            bid_text,
            insert=(margin, bid_text_y),
            fill=text_color,
            font_size=f"{bid_font}px",
            font_family=font_family,
        )
    )

    # Draw the bid face as a small die next to the text
    if current_bid >= 0:
        text_approx_w = len(bid_text) * bid_font * 0.6
        small_die = bid_h * 0.6
        die_x = margin + text_approx_w + bid_font * 0.3
        die_y = y_cursor + (bid_h - small_die) * 0.5
        _draw_die(dwg, board_g, die_x, die_y, small_die, face,
                  pip_color=color_set.p1_color, stroke=color_set.grid_color)

        # Bidder label
        board_g.add(
            dwg.text(
                bidder_text,
                insert=(die_x + small_die + bid_font * 0.4, bid_text_y),
                fill=text_color,
                font_size=f"{bid_font * 0.8}px",
                font_family=font_family,
                fill_opacity="0.5",
            )
        )

    # Terminal: show "LIAR!" label
    if winner >= 0:
        board_g.add(
            dwg.text(
                "LIAR!",
                insert=(total_w - margin, bid_text_y),
                fill=text_color,
                font_size=f"{bid_font}px",
                font_family=font_family,
                font_weight="bold",
                text_anchor="end",
                fill_opacity="0.8",
            )
        )

    y_cursor += bid_h

    # Separator line
    board_g.add(
        dwg.line(
            start=(margin, y_cursor),
            end=(total_w - margin, y_cursor),
            stroke=color_set.grid_color,
            stroke_width="0.5px",
            stroke_opacity="0.3",
        )
    )
    y_cursor += GRID_SIZE * 0.4

    # --- Player 1 ---
    # Player 1 dice
    for i in range(num_dice):
        dx = margin + i * (die_size + die_gap)
        _draw_die(dwg, board_g, dx, y_cursor, die_size, dice[1, i],
                  pip_color=color_set.p1_color, stroke=color_set.grid_color)
    y_cursor += die_size

    # Label
    p1_label = "Player 1"
    if winner == 1:
        p1_label += "  (winner)"
    elif color == 1 and winner < 0:
        p1_label += "  ←"
    board_g.add(
        dwg.text(
            p1_label,
            insert=(margin, y_cursor + label_h * 0.75),
            fill=text_color,
            font_size=f"{label_h * 0.7}px",
            font_family=font_family,
            fill_opacity="0.6",
        )
    )

    return board_g
