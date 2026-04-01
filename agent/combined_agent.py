# Phase 3
"""
Combined agent: belief model + search + policy averaging.
"""

from shared.types import GameState, Move
from agent.belief import policy_average_search


class CombinedAgent:
    """
    Sequence AI agent that uses determinization and policy averaging.
    Same choose_move interface as RandomAgent.
    """

    def __init__(self, n_samples=3, depth=3, search_fn=None):
        self.n_samples = n_samples
        self.depth = depth

        # Use mock search until Phase 2's minimax is merged
        if search_fn is None:
            self.search_fn = self._mock_search
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

        return max(avg_scores, key=avg_scores.get)

    @staticmethod
    def _mock_search(state, depth, player):
        """Placeholder: returns random scores for every legal move."""
        from game.moves import get_legal_moves
        import random

        moves = get_legal_moves(state, player)
        return {move: random.uniform(-10, 10) for move in moves}
