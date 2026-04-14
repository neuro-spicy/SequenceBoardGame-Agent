"""
agent/nn_evaluator.py — CNN value network for board evaluation.
"""

import numpy as np
import torch
import torch.nn as nn
from shared.types import GameState, Card, RANKS, SUITS, BOARD_SIZE, get_opponents

CARD_INDEX = {}
for i, rank in enumerate(RANKS):
    for j, suit in enumerate(SUITS):
        CARD_INDEX[Card(rank, suit)] = i * len(SUITS) + j

NUM_CARD_TYPES = len(RANKS) * len(SUITS)  # 52


def encode_board(state: GameState, player: int) -> torch.Tensor:
    opponents = get_opponents(player)
    grid = state.chip_grid
    board = torch.zeros(4, BOARD_SIZE, BOARD_SIZE)
    board[0] = torch.from_numpy((grid == player).astype(np.float32))
    board[1] = torch.from_numpy(np.isin(grid, opponents).astype(np.float32))
    board[2] = torch.from_numpy((grid == 0).astype(np.float32))
    board[3] = torch.from_numpy((grid == -1).astype(np.float32))
    return board


def encode_hand(state: GameState, player: int) -> torch.Tensor:
    hand_vec = torch.zeros(NUM_CARD_TYPES)
    for card in state.hands.get(player, []):
        idx = CARD_INDEX.get(card)
        if idx is not None:
            hand_vec[idx] += 1.0
    return hand_vec


class SequenceValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * BOARD_SIZE * BOARD_SIZE + NUM_CARD_TYPES, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, board, hand):
        x = self.conv(board)
        x = x.flatten(start_dim=1)
        x = torch.cat([x, hand], dim=1)
        return self.fc(x)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
