"""
game/win_detection.py — sequence detection and win checking.
"""

from shared.types import GameState, SEQUENCES_TO_WIN, NUM_PLAYERS
import numpy as np
from game.board import ALL_LINES, LINE_ROWS, LINE_COLS


def check_sequences(state: GameState, player: int) -> list[tuple]:
    """
    find all completed sequences of 5 for the given player.
    a sequence is 5 player chips in a line, or 4 chips + a corner (wild).
    returns a list of tuples, each containing 5 (row, col) positions.
    """
    line_chips = state.chip_grid[LINE_ROWS, LINE_COLS]  # shape (192, 5)
    match = (line_chips == player) | (line_chips == -1)  # True where chip matches
    complete_mask = match.all(axis=1)                    # True where all 5 match

    completed = []
    for i in np.where(complete_mask)[0]:
        completed.append(ALL_LINES[i])
    return completed


def check_winner(state: GameState) -> int:
    """
    determine if any player has won.
    uses sequence_counts tracked by apply_move, which enforces the
    no-overlap rule: positions in a completed sequence cannot contribute
    to a new one.
    returns the winning player number, or 0 if nobody has won yet.
    """
    required = SEQUENCES_TO_WIN.get(NUM_PLAYERS, 2)
    for player in range(1, NUM_PLAYERS + 1):
        if state.sequence_counts.get(player, 0) >= required:
            return player
    return 0
