# Phase 3
"""
Combined agent: belief model + search + policy averaging.
"""

from shared.types import GameState, Move
from agent.belief import policy_average_search


from agent.search import minimax_search

class CombinedAgent:
    """
    Sequence AI agent that uses determinization and policy averaging.
    Same choose_move interface as RandomAgent.
    """

    def __init__(self, n_samples=3, depth=3, search_fn=None):
        self.n_samples = n_samples
        self.depth = depth

        # Default to real minimax_search
        if search_fn is None:
            self.search_fn = minimax_search
        else:
            self.search_fn = search_fn

    def choose_move(self, state):
        """Pick the move with the highest average score across samples."""
        player = state.current_player

        avg_scores = policy_average_search(
            state, player, self.search_fn, self.n_samples, self.depth,
        )

        if not avg_scores:
            return None

        # Return the move with the highest average score
        return max(avg_scores, key=avg_scores.get)
