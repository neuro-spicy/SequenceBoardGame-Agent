"""
game/board.py — fixed 10x10 board layout and precomputed lookups.

exports:
     BOARD_LAYOUT       - the fixed 10x10 grid
     CARD_TO_POSITIONS  - card -> list of (row, col)
     POSITION_TO_CARD   - (row, col) -> card (96 entries)
     ALL_LINES          - every possible 5-in-a-row (192 total)
     validate_board_layout()  - checks the layout is correct
"""

from collections import Counter
from shared.types import Card
import numpy as np


# None means free corner — every non-jack card appears exactly twice

BOARD_LAYOUT = [
    # row 0
    [
        None,
        Card("2", "spades"),
        Card("3", "spades"),
        Card("4", "spades"),
        Card("5", "spades"),
        Card("6", "spades"),
        Card("7", "spades"),
        Card("8", "spades"),
        Card("9", "spades"),
        None,
    ],
    # row 1
    [
        Card("6", "clubs"),
        Card("5", "clubs"),
        Card("4", "clubs"),
        Card("3", "clubs"),
        Card("2", "clubs"),
        Card("A", "hearts"),
        Card("K", "hearts"),
        Card("Q", "hearts"),
        Card("10", "hearts"),
        Card("10", "spades"),
    ],
    # row 2
    [
        Card("7", "clubs"),
        Card("A", "spades"),
        Card("2", "diamonds"),
        Card("3", "diamonds"),
        Card("4", "diamonds"),
        Card("5", "diamonds"),
        Card("6", "diamonds"),
        Card("7", "diamonds"),
        Card("9", "hearts"),
        Card("Q", "spades"),
    ],
    # row 3
    [
        Card("8", "clubs"),
        Card("K", "spades"),
        Card("6", "clubs"),
        Card("5", "clubs"),
        Card("4", "clubs"),
        Card("3", "clubs"),
        Card("2", "clubs"),
        Card("8", "diamonds"),
        Card("8", "hearts"),
        Card("K", "spades"),
    ],
    # row 4
    [
        Card("9", "clubs"),
        Card("Q", "spades"),
        Card("7", "clubs"),
        Card("6", "hearts"),
        Card("5", "hearts"),
        Card("4", "hearts"),
        Card("A", "hearts"),
        Card("9", "diamonds"),
        Card("7", "hearts"),
        Card("A", "spades"),
    ],
    # row 5
    [
        Card("10", "clubs"),
        Card("10", "spades"),
        Card("8", "clubs"),
        Card("7", "hearts"),
        Card("2", "hearts"),
        Card("3", "hearts"),
        Card("K", "hearts"),
        Card("10", "diamonds"),
        Card("6", "hearts"),
        Card("2", "diamonds"),
    ],
    # row 6
    [
        Card("Q", "clubs"),
        Card("9", "spades"),
        Card("9", "clubs"),
        Card("8", "hearts"),
        Card("9", "hearts"),
        Card("10", "hearts"),
        Card("Q", "hearts"),
        Card("Q", "diamonds"),
        Card("5", "hearts"),
        Card("3", "diamonds"),
    ],
    # row 7
    [
        Card("K", "clubs"),
        Card("8", "spades"),
        Card("10", "clubs"),
        Card("Q", "clubs"),
        Card("K", "clubs"),
        Card("A", "clubs"),
        Card("A", "diamonds"),
        Card("K", "diamonds"),
        Card("4", "hearts"),
        Card("4", "diamonds"),
    ],
    # row 8
    [
        Card("A", "clubs"),
        Card("7", "spades"),
        Card("6", "spades"),
        Card("5", "spades"),
        Card("4", "spades"),
        Card("3", "spades"),
        Card("2", "spades"),
        Card("2", "hearts"),
        Card("3", "hearts"),
        Card("5", "diamonds"),
    ],
    # row 9
    [
        None,
        Card("A", "diamonds"),
        Card("K", "diamonds"),
        Card("Q", "diamonds"),
        Card("10", "diamonds"),
        Card("9", "diamonds"),
        Card("8", "diamonds"),
        Card("7", "diamonds"),
        Card("6", "diamonds"),
        None,
    ],
]


def _build_lookups():
    """create two lookup dicts from the board layout.
    card_to_pos:  find which two cells a card sits on.
    pos_to_card:  find which card is on a given cell.
    """
    card_to_pos = {}
    pos_to_card = {}

    for r in range(10):
        for c in range(10):
            card = BOARD_LAYOUT[r][c]
            if card is not None:
                pos_to_card[(r, c)] = card
                if card not in card_to_pos:
                    card_to_pos[card] = []
                card_to_pos[card].append((r, c))

    return card_to_pos, pos_to_card


CARD_TO_POSITIONS, POSITION_TO_CARD = _build_lookups()


def _build_all_lines():
    """
    compute every possible sequence of 5 on the 10x10 board.

    checks 4 directions from every starting cell:
        horizontal:       (r, c) to (r, c+4)
        vertical:         (r, c) to (r+4, c)
        diagonal right:   (r, c) to (r+4, c+4)
        diagonal left:    (r, c) to (r+4, c-4)
    """
    lines = []

    for r in range(10):
        for c in range(10):

            # horizontal
            if c + 4 <= 9:
                line = ((r, c), (r, c+1), (r, c+2), (r, c+3), (r, c+4))
                lines.append(line)

            # vertical
            if r + 4 <= 9:
                line = ((r, c), (r+1, c), (r+2, c), (r+3, c), (r+4, c))
                lines.append(line)

            # diagonal down-right
            if r + 4 <= 9 and c + 4 <= 9:
                line = ((r, c), (r+1, c+1), (r+2, c+2), (r+3, c+3), (r+4, c+4))
                lines.append(line)

            # diagonal down-left
            if r + 4 <= 9 and c - 4 >= 0:
                line = ((r, c), (r+1, c-1), (r+2, c-2), (r+3, c-3), (r+4, c-4))
                lines.append(line)

    return lines


ALL_LINES = _build_all_lines()


def validate_board_layout():
    """
    checks:
        - exactly 4 None corners at (0,0), (0,9), (9,0), (9,9)
        - no jacks on the board
        - every other card appears exactly twice
        - 96 card cells total
    """
    card_count = Counter()
    none_count = 0

    for r in range(10):
        for c in range(10):
            card = BOARD_LAYOUT[r][c]
            if card is None:
                none_count += 1
            else:
                card_count[card] += 1

    # check corners
    assert none_count == 4, f"Expected 4 corners, got {none_count}"

    # check no jacks
    for card in card_count:
        assert card.rank != "J", f"Jack found on board: {card}"

    # check each card appears exactly twice
    for card, count in card_count.items():
        assert count == 2, f"{card} appears {count} times, expected 2"

    # check total
    assert sum(card_count.values()) == 96, "Expected 96 card cells"

    print("Board layout validation PASSED!")


# precomputed numpy arrays for fast line evaluation
# LINE_ROWS[i] gives the 5 row indices for line i
# LINE_COLS[i] gives the 5 col indices for line i
# used by the heuristic to evaluate all 192 lines at once

LINE_ROWS = np.array([[pos[0] for pos in line] for line in ALL_LINES],
                      dtype=np.int32)  # shape (192, 5)
LINE_COLS = np.array([[pos[1] for pos in line] for line in ALL_LINES],
                      dtype=np.int32)  # shape (192, 5)

# precomputed position values: how many lines pass through each cell
POSITION_VALUES = np.zeros((10, 10), dtype=np.float64)
for line in ALL_LINES:
    for r, c in line:
        POSITION_VALUES[r, c] += 1.0
