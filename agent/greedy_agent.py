"""
agent/greedy_agent.py — Greedy heuristic agent.

Picks the move with the best immediate heuristic score.
No search, no belief model, no lookahead.
Stronger than random, weaker than the combined agent.
"""

from shared.types import GameState, Move
from game.moves import get_legal_moves
from game.game_loop import apply_move
from agent.heuristic import evaluate


class GreedyAgent:
    """
    Same choose_move interface as RandomAgent and CombinedAgent.
    Can be plugged into play_game and run_tournament directly.
    """
    
    def choose_move(self, state: GameState) -> Move:
        player = state.current_player
        moves = get_legal_moves(state, player)
        
        if not moves:
            return None
        
        best_move = None
        best_score = float('-inf')
        
        for move in moves:
            # Try the move on a copy
            child = state.copy()
            apply_move(child, move)
            
            # Score the resulting state from our perspective
            score = evaluate(child, player)
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
