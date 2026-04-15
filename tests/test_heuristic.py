"""tests for agent/heuristic.py"""

from game.game_loop import new_game
from agent.heuristic import evaluate, DEFAULT_WEIGHTS


def test_evaluate_returns_float():
    state = new_game()
    score = evaluate(state, player=1)
    assert isinstance(score, float)


def test_evaluate_both_players_at_start():
    state = new_game()
    # Both players should get valid numeric scores at game start
    s1 = evaluate(state, player=1)
    s2 = evaluate(state, player=2)
    assert isinstance(s1, float)
    assert isinstance(s2, float)


def test_more_chips_gives_higher_score():
    state = new_game()
    baseline = evaluate(state, player=1)

    state.chip_grid[3, 3] = 1
    state.chip_grid[3, 4] = 1
    state.chip_grid[3, 5] = 1
    improved = evaluate(state, player=1)

    assert improved > baseline


def test_opponent_chips_lower_score():
    state = new_game()
    baseline = evaluate(state, player=1)

    # Add a near-complete line for opponent
    for c in range(4):
        state.chip_grid[5, c] = 2
    threatened = evaluate(state, player=1)

    assert threatened < baseline


def test_custom_weights_respected():
    state = new_game()
    state.chip_grid[3, 3] = 1

    # Boost sequence_progress weight; score should go up
    low_w = {"sequence_progress": 0.1, "opponent_threat": 1.2,
             "board_control": 0.3, "jack_utility": 0.5}
    high_w = {"sequence_progress": 5.0, "opponent_threat": 1.2,
              "board_control": 0.3, "jack_utility": 0.5}

    score_low = evaluate(state, player=1, weights=low_w)
    score_high = evaluate(state, player=1, weights=high_w)
    assert score_high > score_low


def test_default_weights_keys():
    expected = {"sequence_progress", "opponent_threat", "board_control", "jack_utility"}
    assert set(DEFAULT_WEIGHTS.keys()) == expected
