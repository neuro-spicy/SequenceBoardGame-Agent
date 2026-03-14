"""
Functions to find completed sequences on the board
Checks to determine if any player has won.
"""

from shared.types import GameState, SEQUENCES_TO_WIN, NUM_PLAYERS
from game.board import ALL_LINES


def check_sequences(state: GameState, player: int) -> list[tuple]:
    """
    Find all completed Sequence of 5 for the given player.
    5 player's chip or a corner wild -1 with 4 player's chip.
    Returns 5  sequence positions.
    """
    completed = []
    for line in ALL_LINES:
        all_match = True
        for r, c in line:
            chip = state.get_chip(r, c)
            if chip != player and chip != -1:
                all_match = False
                break
        if all_match:
            completed.append(line)
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
