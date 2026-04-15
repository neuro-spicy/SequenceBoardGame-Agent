"""tests for game/deck.py"""

from collections import Counter
from game.deck import build_full_deck
from shared.types import Card, RANKS, SUITS


def test_deck_has_104_cards():
    deck = build_full_deck()
    assert len(deck) == 104


def test_deck_is_double_deck():
    deck = build_full_deck()
    counts = Counter(deck)
    for rank in RANKS:
        for suit in SUITS:
            card = Card(rank, suit)
            assert counts[card] == 2, f"{card} appears {counts[card]} times, expected 2"


def test_deck_is_shuffled():
    # Two separately built decks should not be identical (astronomically unlikely)
    deck1 = build_full_deck()
    deck2 = build_full_deck()
    assert deck1 != deck2, "Two freshly built decks are identical — shuffle missing?"


def test_deck_contains_only_valid_cards():
    deck = build_full_deck()
    valid_ranks = set(RANKS)
    valid_suits = set(SUITS)
    for card in deck:
        assert card.rank in valid_ranks
        assert card.suit in valid_suits
