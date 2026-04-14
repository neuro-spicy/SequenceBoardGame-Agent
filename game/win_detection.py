
"""
Phase 1:
Functions to find completed sequences on the board
Checks to determine if any player has won.
"""

from shared.types import GameState, SEQUENCES_TO_WIN, NUM_PLAYERS
import numpy as np
from game.board import ALL_LINES, LINE_ROWS, LINE_COLS


def check_sequences(state: GameState, player: int) -> list[tuple]:
    """
    Find all completed Sequence of 5 for the given player.
    5 player's chip or a corner wild -1 with 4 player's chip.
    Returns 5  sequence positions.
    """
    line_chips = state.chip_grid[LINE_ROWS, LINE_COLS]  # (192, 5)
    match = (line_chips == player) | (line_chips == -1)  # bool (192, 5)
    complete_mask = match.all(axis=1)  # (192,) — True where all 5 match
    
    completed = []
    for i in np.where(complete_mask)[0]:
        completed.append(ALL_LINES[i])
    return completed


def check_winner(state: GameState) -> int:
    """
    Determines if any player has won.
    Checks each player's completed sequences against the required count.
    Returns the winning player number or 0 if no one has won yet.
    """
    required = SEQUENCES_TO_WIN.get(NUM_PLAYERS, 2)
    for player in range(1, NUM_PLAYERS + 1):
        sequences = check_sequences(state, player)
        if len(sequences) >= required:
            return player
    return 0
