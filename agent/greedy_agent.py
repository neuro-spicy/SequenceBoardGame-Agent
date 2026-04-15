"""
agent/greedy_agent.py — greedy heuristic agent.

picks the move with the best immediate heuristic score.
no search, no belief model, no lookahead.
stronger than random, weaker than the combined agent.
"""

from shared.types import GameState, Move
from game.moves import get_legal_moves
from game.game_loop import apply_move
from agent.heuristic import evaluate


class GreedyAgent:
    """
    same choose_move interface as RandomAgent and CombinedAgent.
    can be plugged into play_game and run_tournament directly.
    """

    def choose_move(self, state: GameState) -> Move:
        player = state.current_player
        moves = get_legal_moves(state, player)

        if not moves:
            return None

        best_move = None
        best_score = float('-inf')

        for move in moves:
            # try the move on a copy and score the result
            child = state.copy()
            apply_move(child, move)
            score = evaluate(child, player)

            if score > best_score:
                best_score = score
                best_move = move

        return best_move
