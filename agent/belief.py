"""
agent/belief.py — belief model: card counting, sampling, determinization, and policy averaging.
"""

import random
from collections import Counter, defaultdict

from shared.types import Card, GameState, RANKS, SUITS, get_opponents


def get_unknown_pool(state: GameState, player: int) -> list[Card]:
    """
    return all cards whose location is unknown to the given player.
    unknown = full double deck minus our hand minus the discard pile.
    """
    # build the complete 104-card double deck
    single = [Card(rank, suit) for rank in RANKS for suit in SUITS]
    pool = Counter(single * 2)

    # subtract cards we can see
    for card in state.hands.get(player, []):
        pool[card] -= 1

    for card in state.discard_pile:
        pool[card] -= 1

    # no card count should go negative
    for card, count in pool.items():
        assert count >= 0, f"Negative count for {card}: {count}"

    # flatten back to a list (duplicates preserved)
    unknown = []
    for card, count in pool.items():
        for _ in range(count):
            unknown.append(card)

    return unknown


def sample_opponent_hand(unknown_pool: list[Card], hand_size: int) -> list[Card]:
    """
    sample a random hand from the unknown pool without replacement.
    cards with more copies in the pool are proportionally more likely.
    """
    if hand_size > len(unknown_pool):
        hand_size = len(unknown_pool)

    return random.sample(unknown_pool, hand_size)


def determinize(state: GameState, player: int) -> GameState:
    """
    create a fully observable copy of the state by filling in
    the opponent's hand with a plausible sample from the unknown pool.
    """
    opponents = get_opponents(player)
    unknown_pool = get_unknown_pool(state, player)
    det_state = state.copy()

    # sample a hand for each opponent, removing drawn cards from the pool
    remaining = list(unknown_pool)
    for opp in opponents:
        opp_hand_size = len(state.hands.get(opp, []))
        sampled = sample_opponent_hand(remaining, opp_hand_size)
        det_state.hands[opp] = sampled

        for card in sampled:
            remaining.remove(card)

    # leftover pool becomes the deck
    random.shuffle(remaining)
    det_state.deck = remaining

    return det_state


def policy_average_search(state, player, search_fn, n_samples=3, depth=3):
    """
    determinize n times, run search on each, and average the move scores.
    returns a dict mapping each legal Move to its average score across samples.
    """
    move_total = defaultdict(float)
    move_count = defaultdict(int)

    for _ in range(n_samples):
        det_state = determinize(state, player)
        move_scores = search_fn(det_state, depth, player)

        for move, score in move_scores.items():
            move_total[move] += score
            move_count[move] += 1

    # average each move over however many samples included it
    avg_scores = {}
    for move in move_total:
        avg_scores[move] = move_total[move] / move_count[move]

    return avg_scores
