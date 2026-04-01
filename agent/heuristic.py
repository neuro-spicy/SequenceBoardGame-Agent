"""
Phase 2: Heuristic Evaluation

Scores a board state from the perspective of a given player.
"""

from shared.types import GameState, get_opponents, is_one_eyed_jack, is_two_eyed_jack
from game.board import ALL_LINES

DEFAULT_WEIGHTS = {
    "sequence_progress": 1.0,
    "opponent_threat": 1.2,
    "board_control": 0.3,
    "jack_utility": 0.5,
}

def _build_position_values() -> dict[tuple[int, int], int]:
    """
    Precompute a map of how many 5-in-a-row lines pass through each cell.
    Center cells participate in many lines (high value), edge cells in fewer.
    """
    values: dict[tuple[int, int], int] = {}
    for line in ALL_LINES:
        for pos in line:
            values[pos] = values.get(pos, 0) + 1
    return values

POSITION_VALUES = _build_position_values()

def _score_sequence_progress(state: GameState, player: int) -> float:
    """Score sequence progress based on how many chips are in each valid line."""
    SCORES = {0: 0, 1: 1, 2: 5, 3: 25, 4: 125, 5: 1000}
    total = 0
    opponents = get_opponents(player)
    
    for line in ALL_LINES:
        friendly = 0
        blocked = False
        for r, c in line:
            chip = state.get_chip(r, c)
            if chip == player or chip == -1:  # our chip or corner
                friendly += 1
            elif chip in opponents:
                blocked = True
                break
            # else: empty, don't count
        
        if not blocked:
            total += SCORES.get(friendly, 0)
    
    return float(total)

def _score_opponent_threat(state: GameState, player: int) -> float:
    """Score each opponent's progress and return the maximum threat."""
    opponents = get_opponents(player)
    max_threat = 0.0
    for opp in opponents:
        threat = _score_sequence_progress(state, opp)
        max_threat = max(max_threat, threat)
    return max_threat

def _score_board_control(state: GameState, player: int) -> float:
    """Score board control based on the value of cells occupied by the player."""
    total = 0
    for pos, value in POSITION_VALUES.items():
        r, c = pos
        chip = state.get_chip(r, c)
        if chip == player:
            total += value
    return float(total)

def _score_jack_utility(state: GameState, player: int) -> float:
    """Score utility based on how valuable the Jacks in the player's hand are."""
    hand = state.hands.get(player, [])
    score = 0.0
    opponents = get_opponents(player)
    
    for card in hand:
        if is_two_eyed_jack(card):
            # Valuable when we have 4-in-a-row lines
            score += 10
        elif is_one_eyed_jack(card):
            # Valuable when opponent has threatening lines
            has_threat = False
            for line in ALL_LINES:
                opp_count = 0
                for r, c in line:
                    chip = state.get_chip(r, c)
                    if chip in opponents or chip == -1:
                        opp_count += 1
                if opp_count >= 4:
                    has_threat = True
                    break
            score += 15 if has_threat else 5
    
    return score

import typing

def evaluate(state: GameState, player: int, weights: typing.Optional[dict] = None) -> float:
    """
    Score a board state from the perspective of `player`.
    
    Positive = good for `player`
    Negative = bad for `player`
    Higher magnitude = more decisive advantage
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    
    progress = _score_sequence_progress(state, player)
    threat = _score_opponent_threat(state, player)
    control = _score_board_control(state, player)
    jack = _score_jack_utility(state, player)
    
    score = (
        weights["sequence_progress"] * progress
        - weights["opponent_threat"] * threat
        + weights["board_control"] * control
        + weights["jack_utility"] * jack
    )
    
    return score
