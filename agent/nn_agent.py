"""
agent/nn_agent.py — Agent that uses neural network evaluation.

Same interface as CombinedAgent and RandomAgent.
Uses belief model + determinization + minimax, but replaces the
hand-crafted heuristic with a trained neural network.
"""

from shared.types import GameState, Move
from agent.belief import policy_average_search
from agent.search import minimax_search_with_eval
from agent.nn_evaluator import nn_evaluate, load_model


class NNAgent:
    """
    Neural network-based Sequence agent.
    
    Uses the same belief + determinization + policy averaging pipeline 
    as CombinedAgent, but evaluates leaf nodes with a neural network 
    instead of the hand-crafted heuristic.
    """
    
    def __init__(
        self, 
        model_path: str = "training/models/value_net.pt",
        n_samples: int = 3, 
        depth: int = 2
    ):
        self.n_samples = n_samples
        self.depth = depth
        
        # Load the trained model
        load_model(model_path)
    
    def choose_move(self, state: GameState) -> Move:
        player = state.current_player
        
        def nn_search(s, d, p):
            return minimax_search_with_eval(s, d, p, nn_evaluate)
        
        avg_scores = policy_average_search(
            state, player, nn_search,
            self.n_samples, self.depth,
        )
        
        if not avg_scores:
            return None
        
        return max(avg_scores, key=avg_scores.get)
