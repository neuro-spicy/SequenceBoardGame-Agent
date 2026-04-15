"""tests for agent/search.py"""

from game.game_loop import new_game
from game.moves import get_legal_moves
from agent.search import minimax_search, minimax_search_with_eval, _apply_move_for_search
from agent.heuristic import evaluate
from shared.types import Move


def test_minimax_returns_dict():
    state = new_game()
    result = minimax_search(state, depth=1, player=1)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_minimax_keys_are_legal_moves():
    state = new_game()
    legal = set(get_legal_moves(state, 1))
    result = minimax_search(state, depth=1, player=1)
    for move in result:
        assert move in legal


def test_minimax_scores_are_numeric():
    state = new_game()
    result = minimax_search(state, depth=1, player=1)
    for move, score in result.items():
        assert isinstance(score, (int, float))


def test_minimax_depth2_runs():
    state = new_game()
    result = minimax_search(state, depth=2, player=1)
    assert len(result) > 0


def test_minimax_with_eval_uses_custom_fn():
    state = new_game()
    call_count = [0]

    def counting_eval(s, p):
        call_count[0] += 1
        return evaluate(s, p)

    result = minimax_search_with_eval(state, depth=1, player=1, eval_fn=counting_eval)
    assert len(result) > 0
    assert call_count[0] > 0


def test_apply_move_for_search_places_chip():
    state = new_game()
    moves = get_legal_moves(state, 1)
    place_moves = [m for m in moves if m.move_type == "place"]
    assert place_moves

    move = place_moves[0]
    r, c = move.position
    assert int(state.chip_grid[r, c]) == 0

    child = state.copy()
    _apply_move_for_search(child, move)
    assert int(child.chip_grid[r, c]) == 1


def test_apply_move_for_search_switches_player():
    state = new_game()
    assert state.current_player == 1
    moves = get_legal_moves(state, 1)
    child = state.copy()
    _apply_move_for_search(child, moves[0])
    assert child.current_player == 2
