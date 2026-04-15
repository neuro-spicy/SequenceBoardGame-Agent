"""tests for game/board.py"""

from game.board import (
    BOARD_LAYOUT, CARD_TO_POSITIONS, POSITION_TO_CARD,
    ALL_LINES, LINE_ROWS, LINE_COLS, POSITION_VALUES,
    validate_board_layout,
)
from shared.types import Card


def test_validate_board_layout_passes():
    validate_board_layout()


def test_corners_are_none():
    assert BOARD_LAYOUT[0][0] is None
    assert BOARD_LAYOUT[0][9] is None
    assert BOARD_LAYOUT[9][0] is None
    assert BOARD_LAYOUT[9][9] is None


def test_no_jacks_on_board():
    for row in BOARD_LAYOUT:
        for cell in row:
            if cell is not None:
                assert cell.rank != "J", f"Jack found on board: {cell}"


def test_each_card_appears_twice():
    for card, positions in CARD_TO_POSITIONS.items():
        assert len(positions) == 2, f"{card} has {len(positions)} positions"


def test_card_to_positions_and_position_to_card_consistent():
    for card, positions in CARD_TO_POSITIONS.items():
        for pos in positions:
            assert POSITION_TO_CARD[pos] == card


def test_96_card_cells():
    assert len(POSITION_TO_CARD) == 96


def test_all_lines_count():
    # 192 five-in-a-row lines on a 10x10 board
    assert len(ALL_LINES) == 192


def test_line_rows_cols_shape():
    assert LINE_ROWS.shape == (192, 5)
    assert LINE_COLS.shape == (192, 5)


def test_each_line_has_five_cells():
    for line in ALL_LINES:
        assert len(line) == 5


def test_position_values_positive():
    # Every non-corner cell should be on at least one line
    import numpy as np
    corners = {(0, 0), (0, 9), (9, 0), (9, 9)}
    for r in range(10):
        for c in range(10):
            if (r, c) not in corners:
                assert POSITION_VALUES[r, c] > 0
