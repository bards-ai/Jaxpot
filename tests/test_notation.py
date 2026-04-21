"""Tests for Quoridor text notation."""

import pytest

from jaxpot.env.quoridor.notation import (
    _move_target,
    action_to_text,
    canonical_to_absolute,
    format_game_record,
    text_to_action,
)
from jaxpot.env.quoridor.game import NUM_ACTIONS, WALL_SIZE


# ── Helpers ─────────────────────────────────────────────────────────────────


class TestMoveTarget:
    """Low-level _move_target computation."""

    def test_cardinal_north(self):
        # Action 1 (N): row+1
        assert _move_target(1, 0, 4, 8, 4) == (1, 4)

    def test_cardinal_south(self):
        # Action 0 (S): row-1
        assert _move_target(0, 3, 4, 8, 4) == (2, 4)

    def test_cardinal_east(self):
        # Action 2 (E): col+1
        assert _move_target(2, 4, 4, 8, 4) == (4, 5)

    def test_cardinal_west(self):
        # Action 3 (W): col-1
        assert _move_target(3, 4, 4, 8, 4) == (4, 3)

    def test_straight_jump(self):
        # P0 at (3,4), opp at (4,4), JN(5) -> jump over to (5,4)
        assert _move_target(5, 3, 4, 4, 4) == (5, 4)

    def test_diagonal_approach_first(self):
        # P0 at (3,4), opp at (4,4); action 10 (NE: approach=N, slide=E)
        # Opponent is north -> target = opp + E = (4,5)
        assert _move_target(10, 3, 4, 4, 4) == (4, 5)

    def test_diagonal_approach_reversed(self):
        # P0 at (4,3), opp at (4,4); action 10 (NE: approach=N, slide=E)
        # Opponent is east (not north), so reversed: approach=E, slide=N
        # Target = opp + N = (5,4)
        assert _move_target(10, 4, 3, 4, 4) == (5, 4)


# ── action_to_text ──────────────────────────────────────────────────────────


class TestActionToText:
    """Absolute action index → algebraic text."""

    def test_cardinal_move(self):
        # P0 at (0,4) moves N -> (1,4) = "e2"
        assert action_to_text(1, 0, 4, 8, 4) == "e2"

    def test_cardinal_east(self):
        # P0 at (0,4) moves E -> (0,5) = "f1"
        assert action_to_text(2, 0, 4, 8, 4) == "f1"

    def test_straight_jump(self):
        # P0 at (3,4), opp at (4,4), JN(5) -> (5,4) = "e6"
        assert action_to_text(5, 3, 4, 4, 4) == "e6"

    def test_diagonal_jump(self):
        # P0 at (3,4), opp at (4,4), NE(10) -> (4,5) = "f5"
        assert action_to_text(10, 3, 4, 4, 4) == "f5"

    def test_horizontal_wall(self):
        # action 12 + 3*8 + 4 = 40 -> grid (3,4) = "e4h"
        assert action_to_text(40, 0, 0, 0, 0) == "e4h"

    def test_vertical_wall(self):
        # action 76 + 5*8 + 2 = 118 -> grid (5,2) = "c6v"
        assert action_to_text(118, 0, 0, 0, 0) == "c6v"

    def test_first_horizontal_wall(self):
        # action 12 -> grid (0,0) = "a1h"
        assert action_to_text(12, 0, 0, 0, 0) == "a1h"

    def test_last_vertical_wall(self):
        # action 139 -> grid (7,7) = "h8v"
        assert action_to_text(139, 0, 0, 0, 0) == "h8v"

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            action_to_text(-1, 0, 0, 0, 0)
        with pytest.raises(ValueError):
            action_to_text(140, 0, 0, 0, 0)


# ── text_to_action ──────────────────────────────────────────────────────────


class TestTextToAction:
    """Algebraic text → absolute action index."""

    def test_cardinal_move(self):
        # P0 at (0,4), target "e2" = (1,4) -> action 1 (N)
        assert text_to_action("e2", 0, 4, 8, 4) == 1

    def test_straight_jump(self):
        # P0 at (3,4), opp at (4,4), target "e6" = (5,4) -> JN(5)
        assert text_to_action("e6", 3, 4, 4, 4) == 5

    def test_diagonal_jump(self):
        # P0 at (3,4), opp at (4,4), target "f5" = (4,5) -> NE(10)
        assert text_to_action("f5", 3, 4, 4, 4) == 10

    def test_horizontal_wall(self):
        assert text_to_action("e4h", 0, 0, 0, 0) == 40

    def test_vertical_wall(self):
        assert text_to_action("c6v", 0, 0, 0, 0) == 118

    def test_case_insensitive(self):
        assert text_to_action("E4H", 0, 0, 0, 0) == 40
        assert text_to_action("C6V", 0, 0, 0, 0) == 118

    def test_whitespace_stripped(self):
        assert text_to_action("  e4h  ", 0, 0, 0, 0) == 40

    def test_invalid_wall_column_raises(self):
        with pytest.raises(ValueError):
            text_to_action("z4h", 0, 0, 0, 0)

    def test_wall_row_out_of_range_raises(self):
        with pytest.raises(ValueError):
            text_to_action("a0h", 0, 0, 0, 0)
        with pytest.raises(ValueError):
            text_to_action("a9h", 0, 0, 0, 0)

    def test_invalid_pawn_column_raises(self):
        with pytest.raises(ValueError):
            text_to_action("z5", 0, 0, 0, 0)

    def test_no_valid_move_raises(self):
        # Target (5,5) unreachable from (0,4) with opp at (8,4)
        with pytest.raises(ValueError):
            text_to_action("f6", 0, 4, 8, 4)


# ── Roundtrips ──────────────────────────────────────────────────────────────


class TestRoundtrips:
    """action_to_text → text_to_action should round-trip."""

    def test_all_walls_roundtrip(self):
        for action in range(12, NUM_ACTIONS):
            text = action_to_text(action, 0, 0, 0, 0)
            assert text_to_action(text, 0, 0, 0, 0) == action

    def test_cardinal_roundtrip(self):
        # P0 at center (4,4), opp far away at (8,0)
        for action in range(4):
            text = action_to_text(action, 4, 4, 8, 0)
            assert text_to_action(text, 4, 4, 8, 0) == action

    def test_jump_roundtrip(self):
        # P0 at (4,4), opp at (5,4) -> JN(5) legal
        text = action_to_text(5, 4, 4, 5, 4)
        assert text_to_action(text, 4, 4, 5, 4) == 5

    def test_diagonal_roundtrip(self):
        # P0 at (7,4), opp at (8,4) -> diagonals NE(10), NW(11) when straight jump blocked by edge
        for action in (10, 11):
            text = action_to_text(action, 7, 4, 8, 4)
            assert text_to_action(text, 7, 4, 8, 4) == action


# ── canonical_to_absolute ──────────────────────────────────────────────────


class TestCanonicalToAbsolute:
    """Perspective conversion for the env's canonical action space."""

    def test_color0_is_identity(self):
        for a in range(NUM_ACTIONS):
            assert canonical_to_absolute(a, 0) == a

    def test_double_flip_is_identity(self):
        for a in range(NUM_ACTIONS):
            assert canonical_to_absolute(canonical_to_absolute(a, 1), 1) == a

    def test_cardinal_flip(self):
        # S(0) <-> N(1), E(2) <-> W(3)
        assert canonical_to_absolute(0, 1) == 1
        assert canonical_to_absolute(1, 1) == 0
        assert canonical_to_absolute(2, 1) == 3
        assert canonical_to_absolute(3, 1) == 2

    def test_jump_flip(self):
        assert canonical_to_absolute(4, 1) == 5
        assert canonical_to_absolute(5, 1) == 4

    def test_diagonal_flip(self):
        # SE(8) <-> NW(11), SW(9) <-> NE(10)
        assert canonical_to_absolute(8, 1) == 11
        assert canonical_to_absolute(11, 1) == 8
        assert canonical_to_absolute(9, 1) == 10
        assert canonical_to_absolute(10, 1) == 9

    def test_horizontal_wall_flip(self):
        # Wall index i -> 63-i within the 64-slot block
        # action 12 (h-wall at idx 0) -> action 75 (h-wall at idx 63)
        assert canonical_to_absolute(12, 1) == 75
        assert canonical_to_absolute(75, 1) == 12

    def test_vertical_wall_flip(self):
        assert canonical_to_absolute(76, 1) == 139
        assert canonical_to_absolute(139, 1) == 76


# ── format_game_record ─────────────────────────────────────────────────────


class TestFormatGameRecord:
    """PGN-like game record generation."""

    def test_header_tags(self):
        record = format_game_record([], "Agent", "Random", "1/2-1/2", 0)
        assert '[Game "Quoridor"]' in record
        assert '[P0 "Agent"]' in record
        assert '[P1 "Random"]' in record
        assert '[Result "1/2-1/2"]' in record
        assert '[FirstPlayer "0"]' in record

    def test_empty_game(self):
        record = format_game_record([], "A", "B", "1/2-1/2", 0)
        # Should have headers only, no move lines
        lines = record.strip().split("\n")
        assert lines[-1] == '[FirstPlayer "0"]'

    def test_opening_moves(self):
        # Both players play canonical N(1):
        # Color 0 (first_player=0): canonical N=abs N, P0 (0,4)->(1,4) = "e2"
        # Color 1 (player=1): canonical N=abs S (flipped), P1 (8,4)->(7,4) = "e8"
        actions = [(0, 1), (1, 1)]
        record = format_game_record(actions, "A", "B", "1-0", 0)
        assert "1. e2 e8" in record

    def test_position_tracking(self):
        # Four moves: both keep playing canonical N
        # Move 1: P0 (0,4)->(1,4)=e2, P1 (8,4)->(7,4)=e8
        # Move 2: P0 (1,4)->(2,4)=e3, P1 (7,4)->(6,4)=e7
        actions = [(0, 1), (1, 1), (0, 1), (1, 1)]
        record = format_game_record(actions, "A", "B", "1-0", 0)
        assert "1. e2 e8" in record
        assert "2. e3 e7" in record

    def test_odd_number_of_moves(self):
        # Single move game
        actions = [(0, 1)]
        record = format_game_record(actions, "A", "B", "1-0", 0)
        assert "1. e2" in record

    def test_wall_in_record(self):
        # P0 places a wall, P1 moves
        # Canonical h-wall at idx 0 = action 12 for color 0 = absolute action 12 = a1h
        actions = [(0, 12), (1, 1)]
        record = format_game_record(actions, "A", "B", "1-0", 0)
        assert "1. a1h e8" in record

    def test_first_player_1(self):
        # When first_player=1, player 1 is internal color 0 (no flip)
        # Player 1 plays canonical N(1) = abs N from (0,4) -> (1,4) = e2
        # Player 0 plays canonical N(1) = abs S (color 1 flip) from (8,4) -> (7,4) = e8
        actions = [(1, 1), (0, 1)]
        record = format_game_record(actions, "A", "B", "0-1", 1)
        assert '[FirstPlayer "1"]' in record
        assert "1. e2 e8" in record

    def test_result_strings(self):
        for result in ("1-0", "0-1", "1/2-1/2"):
            record = format_game_record([], "A", "B", result, 0)
            assert f'[Result "{result}"]' in record


# ── Perspective independence ────────────────────────────────────────────────


class TestPerspectiveIndependence:
    """Notation should be the same regardless of which player we view from."""

    def test_same_square_different_player(self):
        # P0 at (0,4) moves N to (1,4) = "e2"
        assert action_to_text(1, 0, 4, 8, 4) == "e2"
        # P1 at (2,4) moves S to (1,4) = also "e2"
        assert action_to_text(0, 2, 4, 8, 0) == "e2"
        # text_to_action: "e2" resolves to the correct action for each position
        assert text_to_action("e2", 0, 4, 8, 4) == 1  # cardinal N from (0,4)
        assert text_to_action("e2", 2, 4, 8, 0) == 0  # cardinal S from (2,4)

    def test_wall_always_same_text(self):
        # Wall e4h is always action 40 regardless of who placed it
        assert text_to_action("e4h", 0, 4, 8, 4) == 40
        assert text_to_action("e4h", 8, 4, 0, 4) == 40
        assert action_to_text(40, 0, 0, 0, 0) == "e4h"
        assert action_to_text(40, 8, 4, 0, 4) == "e4h"
