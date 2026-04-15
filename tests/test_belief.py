"""tests for agent/belief.py"""

from game.game_loop import new_game
from agent.belief import get_unknown_pool, sample_opponent_hand, determinize, policy_average_search
from agent.search import minimax_search
from shared.types import HAND_SIZE, Card


def test_unknown_pool_no_negatives():
    state = new_game()
    pool = get_unknown_pool(state, player=1)
    assert len(pool) > 0


def test_unknown_pool_excludes_our_hand():
    state = new_game()
    pool = get_unknown_pool(state, player=1)
    from collections import Counter
    pool_counts = Counter(pool)
    hand_counts = Counter(state.hands[1])
    # No card should appear in pool more times than (2 - hand_count)
    for card, count in hand_counts.items():
        assert pool_counts.get(card, 0) <= 2 - count


def test_unknown_pool_total_cards():
    state = new_game()
    pool = get_unknown_pool(state, player=1)
    # Total known = our hand (7) + discard (0) + opponent hand (unknown but drawn)
    # Pool + our hand + discard + opponent's actual drawn cards = 104
    # At start: pool = 104 - 7 (our hand) - 0 (discard) = 97... but opponent holds 7
    # Pool represents everything we DON'T know, which is 104 - 7 (our hand) = 97
    assert len(pool) == 104 - HAND_SIZE


def test_sample_opponent_hand_correct_size():
    state = new_game()
    pool = get_unknown_pool(state, player=1)
    hand = sample_opponent_hand(pool, HAND_SIZE)
    assert len(hand) == HAND_SIZE


def test_sample_opponent_hand_from_pool():
    state = new_game()
    pool = get_unknown_pool(state, player=1)
    from collections import Counter
    pool_counts = Counter(pool)
    hand = sample_opponent_hand(pool, HAND_SIZE)
    hand_counts = Counter(hand)
    for card, count in hand_counts.items():
        assert count <= pool_counts[card]


def test_determinize_fills_opponent_hand():
    state = new_game()
    det = determinize(state, player=1)
    assert len(det.hands[2]) == HAND_SIZE


def test_determinize_preserves_our_hand():
    state = new_game()
    det = determinize(state, player=1)
    assert det.hands[1] == state.hands[1]


def test_determinize_card_conservation():
    state = new_game()
    det = determinize(state, player=1)
    total = (len(det.hands[1]) + len(det.hands[2])
             + len(det.deck) + len(det.discard_pile))
    assert total == 104


def test_policy_average_search_returns_dict():
    state = new_game()
    result = policy_average_search(state, player=1,
                                   search_fn=minimax_search,
                                   n_samples=2, depth=1)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_policy_average_scores_numeric():
    state = new_game()
    result = policy_average_search(state, player=1,
                                   search_fn=minimax_search,
                                   n_samples=2, depth=1)
    for move, score in result.items():
        assert isinstance(score, (int, float))
