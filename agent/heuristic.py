# Phase 2
"""
Vectorized sequence heuristic evaluation.
"""

import numpy as np
from shared.types import GameState, get_opponents, is_one_eyed_jack, is_two_eyed_jack
from game.board import ALL_LINES, LINE_ROWS, LINE_COLS, POSITION_VALUES

DEFAULT_WEIGHTS = {
    "sequence_progress": 1.0,
    "opponent_threat": 1.2,
    "board_control": 0.3,
    "jack_utility": 0.5,
}

SCORE_TABLE = np.array([0, 1, 5, 25, 125, 1000], dtype=np.float64)


def _score_progress(board: np.ndarray, player: int, 
                     opponents: list[int]) -> float:
    line_chips = board[LINE_ROWS, LINE_COLS]  # (192, 5)
    
    friendly = (line_chips == player) | (line_chips == -1)
    
    blocked = np.zeros(len(ALL_LINES), dtype=bool)
    for opp in opponents:
        blocked |= (line_chips == opp).any(axis=1)
    
    counts = friendly.sum(axis=1)  # (192,)
    counts[blocked] = 0
    counts = np.clip(counts, 0, 5)
    
    return float(SCORE_TABLE[counts].sum())


def _score_control(board: np.ndarray, player: int) -> float:
    return float((POSITION_VALUES * (board == player)).sum())


def _score_jacks(state: GameState, player: int,
                  board: np.ndarray, opponents: list[int]) -> float:
    hand = state.hands.get(player, [])
    score = 0.0
    
    has_one_eyed = any(is_one_eyed_jack(c) for c in hand)
    threat = False
    
    if has_one_eyed:
        line_chips = board[LINE_ROWS, LINE_COLS]
        for opp in opponents:
            opp_friendly = (line_chips == opp) | (line_chips == -1)
            if (opp_friendly.sum(axis=1) >= 4).any():
                threat = True
                break
    
    for card in hand:
        if is_two_eyed_jack(card):
            score += 10.0
        elif is_one_eyed_jack(card):
            score += 15.0 if threat else 5.0
    
    return score


def evaluate(state: GameState, player: int, weights=None) -> float:
    if weights is None:
        weights = DEFAULT_WEIGHTS
    
    opponents = get_opponents(player)
    board = state.chip_grid  # already numpy, no conversion needed
    
    progress = _score_progress(board, player, opponents)
    threat = _score_progress(board, opponents[0], [player])
    control = _score_control(board, player)
    jack = _score_jacks(state, player, board, opponents)
    
    return (
        weights["sequence_progress"] * progress
        - weights["opponent_threat"] * threat
        + weights["board_control"] * control
        + weights["jack_utility"] * jack
    )
