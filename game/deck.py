"""
game/deck.py — deck construction for Sequence.

a Sequence deck is two copies of a standard 52-card deck, shuffled.
"""

import random
from shared.types import Card, RANKS, SUITS


def build_full_deck() -> list[Card]:
    """build and return a shuffled 104-card double deck."""
    single_deck = []
    for rank in RANKS:
        for suit in SUITS:
            single_deck.append(Card(rank, suit))

    double_deck = single_deck + single_deck
    random.shuffle(double_deck)
    return double_deck
