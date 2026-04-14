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
        n_samples: int = 3,
        depth: int = 2,
    ):
        self.n_samples = n_samples
        self.depth = depth
        self.device = torch.device("cpu")  # CPU faster for single samples
        self._cache = {}

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

        # JIT compile for faster inference
        try:
            self.model = torch.jit.trace(
                self.model,
                (torch.randn(1, 4, 10, 10), torch.randn(1, 52))
            )
        except Exception:
            pass  # fall back to normal model if tracing fails

        print(f"Loaded NN model from {model_path} (CPU, JIT compiled)")

    def _nn_evaluate(self, state: GameState, player: int) -> float:
        key = (state.chip_grid.tobytes(), player)
        if key in self._cache:
            return self._cache[key]

        with torch.no_grad():
            board = encode_board(state, player).unsqueeze(0)
            hand = encode_hand(state, player).unsqueeze(0)
            normalized = self.model(board, hand).item()

        score = normalized * self.score_std + self.score_mean
        self._cache[key] = score
        return score

    def choose_move(self, state: GameState) -> Move:
        self._cache = {}
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
