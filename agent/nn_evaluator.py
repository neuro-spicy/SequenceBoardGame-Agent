"""
agent/nn_evaluator.py — Neural network board evaluation.
"""

import numpy as np
import torch
import torch.nn as nn
from shared.types import GameState, Card, RANKS, SUITS, BOARD_SIZE, get_opponents


# Build a fixed mapping: Card → index (0 to 51)
CARD_INDEX = {}
for i, rank in enumerate(RANKS):
    for j, suit in enumerate(SUITS):
        CARD_INDEX[Card(rank, suit)] = i * len(SUITS) + j

NUM_CARD_TYPES = len(RANKS) * len(SUITS)  # 52


def encode_board(state: GameState, player: int) -> torch.Tensor:
    """
    Encode the board as a 4×10×10 float tensor.
    
    Channel 0: our chips
    Channel 1: opponent chips
    Channel 2: empty cells
    Channel 3: corner wilds
    
    Uses numpy directly since chip_grid is already numpy.
    """
    opponents = get_opponents(player)
    grid = state.chip_grid  # already numpy int32
    
    board = torch.zeros(4, BOARD_SIZE, BOARD_SIZE)
    board[0] = torch.from_numpy((grid == player).astype(np.float32))
    board[1] = torch.from_numpy(
        np.isin(grid, opponents).astype(np.float32)
    )
    board[2] = torch.from_numpy((grid == 0).astype(np.float32))
    board[3] = torch.from_numpy((grid == -1).astype(np.float32))
    
    return board


def encode_hand(state: GameState, player: int) -> torch.Tensor:
    """
    Encode the player's hand as a 52-length float vector.
    Each position = count of that card in hand (0, 1, or 2).
    """
    hand_vec = torch.zeros(NUM_CARD_TYPES)
    for card in state.hands.get(player, []):
        idx = CARD_INDEX.get(card)
        if idx is not None:
            hand_vec[idx] += 1.0
    return hand_vec


def encode_state(state: GameState, player: int) -> torch.Tensor:
    """
    Full state encoding: flatten board channels + hand vector.
    Returns a 1D tensor of size 4*10*10 + 52 = 452.
    """
    board = encode_board(state, player).flatten()  # 400
    hand = encode_hand(state, player)               # 52
    return torch.cat([board, hand])                  # 452


class SequenceValueNet(nn.Module):
    """
    Neural network that predicts game outcome from board state.
    
    Input:  452 features (4×10×10 board + 52 hand)
    Output: single value between -1 and +1
            -1 = losing badly, +1 = winning easily
    
    Architecture: 3 hidden layers with ReLU, tanh output.
    ~50K parameters — small enough to train on CPU in minutes.
    """
    
    def __init__(self, input_size=452, hidden_size=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),  # output between -1 and +1
        )
    
    def forward(self, x):
        return self.network(x)


def get_device():
    """Use MPS (Apple Silicon GPU) if available, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Global model instance (loaded once, used for all evaluations)
_model = None
_device = None


def load_model(path: str = "training/models/value_net.pt"):
    """Load a trained model from disk."""
    global _model, _device
    _device = get_device()
    _model = SequenceValueNet()
    _model.load_state_dict(torch.load(path, map_location=_device))
    _model.to(_device)
    _model.eval()
    print(f"Loaded NN model from {path} on {_device}")


def nn_evaluate(state: GameState, player: int) -> float:
    """
    Evaluate a board state using the neural network.
    
    Same interface as heuristic.evaluate().
    Returns a float score from the perspective of `player`.
    
    Must call load_model() before first use.
    """
    if _model is None:
        raise RuntimeError("Call load_model() before nn_evaluate()")
    
    with torch.no_grad():
        encoded = encode_state(state, player).unsqueeze(0).to(_device)
        value = _model(encoded).item()
    
    # Scale to match roughly the same range as the heuristic
    # tanh output is -1 to 1, scale to -1000 to 1000
    return value * 1000.0
