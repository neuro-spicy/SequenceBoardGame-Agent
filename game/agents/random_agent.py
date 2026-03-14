"""
Baseline random agent for testing and stress testing the game loop.
"""

import random
from game.moves import get_legal_moves

class RandomAgent:
    def choose_move(self, state):
        moves = get_legal_moves(state)
        if not moves:
            return None
        return random.choice(moves)
