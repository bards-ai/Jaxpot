"""Text notation for Quoridor games.

Algebraic target-square notation (perspective-independent, like chess):
  Pawn moves: <col><row> — e.g. ``e5`` means pawn lands on column e, row 5
  Walls:      <col><row><h|v> — e.g. ``e4h`` is a horizontal wall

Columns a-i left to right, rows 1-9 bottom to top (Wikipedia Modern Algebraic).
Row 1 is at the bottom (south), row 9 at the top (north).
Pawn squares: a1-i9.  Wall intersections: a1-h8.
"""

from .game import BOARD_SIZE, NUM_ACTIONS, WALL_SIZE

# Direction offsets matching game.py: S(0), N(1), E(2), W(3)
_DR = [-1, 1, 0, 0]
_DC = [0, 0, 1, -1]

# Diagonal approach/slide combinations matching game.py
_DIAG_COMBOS = [(0, 2), (0, 3), (1, 2), (1, 3)]

# 180-degree rotation for canonical -> absolute (same as _MOVE_FLIP in env.py).
_MOVE_FLIP = [1, 0, 3, 2, 5, 4, 7, 6, 11, 10, 9, 8]

_PAWN_COLS = "abcdefghi"  # 9 columns for pawn squares
_WALL_COLS = "abcdefgh"  # 8 columns for wall intersections


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _move_target(action: int, my_r: int, my_c: int,
                 opp_r: int, opp_c: int) -> tuple[int, int]:
    """Compute target square (row, col) for a movement action (0-11)."""
    if action < 4:  # Cardinal
        return my_r + _DR[action], my_c + _DC[action]
    if action < 8:  # Straight jump
        d = action - 4
        return opp_r + _DR[d], opp_c + _DC[d]
    # Diagonal jump (8-11)
    d = action - 8
    approach, slide = _DIAG_COMBOS[d]
    adj_r = my_r + _DR[approach]
    adj_c = my_c + _DC[approach]
    if adj_r == opp_r and adj_c == opp_c:
        return opp_r + _DR[slide], opp_c + _DC[slide]
    # Reversed: approach via slide direction, slide via approach direction
    return opp_r + _DR[approach], opp_c + _DC[approach]


def _sq(row: int, col: int) -> str:
    """(row, col) -> algebraic square, e.g. (0, 4) -> 'e1'."""
    return f"{_PAWN_COLS[col]}{row + 1}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def action_to_text(action: int, my_r: int, my_c: int,
                   opp_r: int, opp_c: int) -> str:
    """Convert an absolute action index to algebraic text.

    For movement actions (0-11) the pawn positions are needed to compute
    the target square.  Wall actions (12-139) ignore the positions.
    """
    if action < 0 or action >= NUM_ACTIONS:
        raise ValueError(f"Action {action} out of range [0, {NUM_ACTIONS})")

    if action < 12:
        tr, tc = _move_target(action, my_r, my_c, opp_r, opp_c)
        return _sq(tr, tc)

    if action < 76:
        idx = action - 12
        return f"{_WALL_COLS[idx % WALL_SIZE]}{idx // WALL_SIZE + 1}h"

    idx = action - 76
    return f"{_WALL_COLS[idx % WALL_SIZE]}{idx // WALL_SIZE + 1}v"


def text_to_action(text: str, my_r: int, my_c: int,
                   opp_r: int, opp_c: int) -> int:
    """Convert algebraic text to an absolute action index.

    For pawn target squares (e.g. ``"e5"``), positions are required.
    For walls (e.g. ``"e4h"``), positions are ignored.
    """
    t = text.strip().lower()

    # Wall notation: >= 3 chars ending in h/v
    if len(t) >= 3 and t[-1] in ("h", "v"):
        col_ch, wall_type = t[0], t[-1]
        row_num = int(t[1:-1]) - 1
        if col_ch not in _WALL_COLS:
            raise ValueError(f"Invalid wall column '{col_ch}'")
        col = _WALL_COLS.index(col_ch)
        if not 0 <= row_num < WALL_SIZE:
            raise ValueError(f"Wall row out of range [1, {WALL_SIZE}]")
        idx = row_num * WALL_SIZE + col
        return 12 + idx if wall_type == "h" else 76 + idx

    # Pawn target square (e.g. "e5")
    col_ch = t[0]
    if col_ch not in _PAWN_COLS:
        raise ValueError(f"Invalid column '{col_ch}'")
    target_c = _PAWN_COLS.index(col_ch)
    target_r = int(t[1:]) - 1
    if not 0 <= target_r < BOARD_SIZE:
        raise ValueError(f"Row out of range [1, {BOARD_SIZE}]")

    # Cardinal
    for d in range(4):
        if my_r + _DR[d] == target_r and my_c + _DC[d] == target_c:
            return d
    # Straight jump
    for d in range(4):
        adj_r, adj_c = my_r + _DR[d], my_c + _DC[d]
        if adj_r == opp_r and adj_c == opp_c:
            if opp_r + _DR[d] == target_r and opp_c + _DC[d] == target_c:
                return 4 + d
    # Diagonal jump
    for d in range(4):
        approach, slide = _DIAG_COMBOS[d]
        # approach0 + slide0
        if (my_r + _DR[approach] == opp_r and my_c + _DC[approach] == opp_c
                and opp_r + _DR[slide] == target_r and opp_c + _DC[slide] == target_c):
            return 8 + d
        # reversed: approach via slide, slide via approach
        if (my_r + _DR[slide] == opp_r and my_c + _DC[slide] == opp_c
                and opp_r + _DR[approach] == target_r and opp_c + _DC[approach] == target_c):
            return 8 + d

    raise ValueError(
        f"No valid action for target {text} from ({my_r},{my_c}) opp ({opp_r},{opp_c})"
    )


def canonical_to_absolute(action: int, color: int) -> int:
    """Convert a canonical (perspective-normalized) action to absolute coords.

    Color 0 actions are already absolute.  Color 1 actions are rotated 180
    degrees.
    """
    if color == 0:
        return action
    if action < 12:
        return _MOVE_FLIP[action]
    if action < 76:
        return 75 - (action - 12)
    return 139 - (action - 76)


def format_game_record(
    actions: list[tuple[int, int]],
    p0_label: str,
    p1_label: str,
    result: str,
    first_player: int,
) -> str:
    """Format game actions into a PGN-like text record.

    Args:
        actions: List of ``(current_player, canonical_action)`` tuples.
        p0_label: Label for PGX player 0.
        p1_label: Label for PGX player 1.
        result: ``"1-0"`` (P0 wins), ``"0-1"`` (P1 wins), or ``"1/2-1/2"``.
        first_player: PGX player ID that moved first.
    """
    lines = [
        '[Game "Quoridor"]',
        f'[P0 "{p0_label}"]',
        f'[P1 "{p1_label}"]',
        f'[Result "{result}"]',
        f'[FirstPlayer "{first_player}"]',
        "",
    ]

    # Track pawn positions: internal color 0 starts at (0,4), color 1 at (8,4)
    pos = {0: (0, 4), 1: (8, 4)}

    text_actions = []
    for player, canonical_action in actions:
        internal_color = 0 if player == first_player else 1
        abs_action = canonical_to_absolute(canonical_action, internal_color)

        my_r, my_c = pos[internal_color]
        opp_r, opp_c = pos[1 - internal_color]

        text_actions.append(action_to_text(abs_action, my_r, my_c, opp_r, opp_c))

        # Update position for movement actions
        if abs_action < 12:
            pos[internal_color] = _move_target(abs_action, my_r, my_c, opp_r, opp_c)

    move_num = 1
    i = 0
    while i < len(text_actions):
        if i + 1 < len(text_actions):
            lines.append(f"{move_num}. {text_actions[i]} {text_actions[i + 1]}")
            i += 2
        else:
            lines.append(f"{move_num}. {text_actions[i]}")
            i += 1
        move_num += 1

    return "\n".join(lines) + "\n"
