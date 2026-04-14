"""
agent/nn_agent.py — Agent using CNN evaluation.
"""

import torch
from shared.types import GameState, Move
from agent.belief import policy_average_search
from agent.search import minimax_search_with_eval
from agent.nn_evaluator import (
    SequenceValueNet, encode_board, encode_hand, get_device
)


class NNAgent:
    def __init__(
        self,
        model_path: str = "training/models/value_net_v2.pt",
        n_samples: int = 5,
        depth: int = 3,
    ):
        self.n_samples = n_samples
        self.depth = depth
        self.device = get_device()

        checkpoint = torch.load(model_path, map_location=self.device)

        self.model = SequenceValueNet()
        if "model_state" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state"])
            self.score_mean = checkpoint.get("score_mean", 0.0)
            self.score_std = checkpoint.get("score_std", 1.0)
        else:
            self.model.load_state_dict(checkpoint)
            self.score_mean = 0.0
            self.score_std = 1.0

        self.model.to(self.device)
        self.model.eval()
        print(f"Loaded NN model from {model_path} on {self.device}")

    def _nn_evaluate(self, state: GameState, player: int) -> float:
        with torch.no_grad():
            board = encode_board(state, player).unsqueeze(0).to(self.device)
            hand = encode_hand(state, player).unsqueeze(0).to(self.device)
            normalized = self.model(board, hand).item()
        return normalized * self.score_std + self.score_mean

    def choose_move(self, state: GameState) -> Move:
        player = state.current_player

        def nn_search(s, d, p):
            return minimax_search_with_eval(s, d, p, self._nn_evaluate)

        avg_scores = policy_average_search(
            state, player, nn_search,
            self.n_samples, self.depth,
        )

        if not avg_scores:
            return None
        return max(avg_scores, key=avg_scores.get)
