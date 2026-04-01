"""
tests/test_integration.py — Full pipeline integration tests.
"""
import time
from shared.types import Card, GameState, HAND_SIZE
from game.game_loop import new_game, play_game, run_tournament
from game.agents.random_agent import RandomAgent
from game.moves import get_legal_moves
from agent.combined_agent import CombinedAgent
from agent.belief import get_unknown_pool, determinize
from agent.search import minimax_search
from agent.heuristic import evaluate


def test_smoke_single_game():
    """Combined agent plays one full game without crashing."""
    ai = CombinedAgent(n_samples=2, depth=1)
    rand = RandomAgent()
    winner = play_game(ai, rand, max_turns=300)
    assert winner in (0, 1, 2)


def test_combined_returns_legal_move():
    """Every move the agent returns must be in the legal moves list."""
    state = new_game()
    ai = CombinedAgent(n_samples=2, depth=1)
    
    for _ in range(20):
        if state.current_player == 1:
            move = ai.choose_move(state)
            legal = get_legal_moves(state, 1)
            assert move in legal, f"Illegal move: {move}"
        # Play a random move to advance the game
        from game.game_loop import apply_move
        moves = get_legal_moves(state)
        if moves:
            import random
            apply_move(state, random.choice(moves))
        else:
            break


def test_determinized_state_is_valid():
    """Determinized states should have correct card counts."""
    state = new_game()
    det = determinize(state, player=1)
    
    # Both hands should be filled
    assert len(det.hands[1]) == HAND_SIZE
    assert len(det.hands[2]) == HAND_SIZE
    
    # Our hand should be unchanged
    assert det.hands[1] == state.hands[1]
    
    # Total cards should be 104
    total = (len(det.hands[1]) + len(det.hands[2]) 
             + len(det.deck) + len(det.discard_pile))
    assert total == 104, f"Card count: {total}, expected 104"


def test_search_on_determinized_state():
    """Minimax search should work on a determinized state."""
    state = new_game()
    det = determinize(state, player=1)
    
    result = minimax_search(det, depth=1, player=1)
    assert isinstance(result, dict)
    assert len(result) > 0
    
    for move, score in result.items():
        assert isinstance(score, float) or isinstance(score, int)


def test_heuristic_prefers_winning():
    """A near-winning state should score much higher than empty."""
    empty = new_game()
    score_empty = evaluate(empty, player=1)
    
    # Create a state with 4-in-a-row for player 1
    winning = new_game()
    winning.set_chip(0, 1, 1)
    winning.set_chip(0, 2, 1)
    winning.set_chip(0, 3, 1)
    winning.set_chip(0, 4, 1)
    score_winning = evaluate(winning, player=1)
    
    assert score_winning > score_empty


def test_ai_beats_random():
    """Combined agent should win more than 50% against random."""
    ai = CombinedAgent(n_samples=2, depth=2) # Keep low depth for quick tests
    rand = RandomAgent()
    results = run_tournament(ai, rand, n_games=20) # Lowered to 20 to avoid waiting an hour
    
    win_rate = results[1] / 20.0
    print(f"AI win rate: {win_rate:.0%}")
    assert win_rate > 0.5, f"AI only won {win_rate:.0%}"


def test_decision_time():
    """Each decision should complete in reasonable time."""
    state = new_game()
    ai = CombinedAgent(n_samples=2, depth=2)
    
    start = time.time()
    move = ai.choose_move(state)
    elapsed = time.time() - start
    
    print(f"First move decision time: {elapsed:.2f}s")
    assert elapsed < 10, f"Decision took {elapsed:.1f}s, too slow"
