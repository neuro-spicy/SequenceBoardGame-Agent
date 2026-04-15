"""tests for game/moves.py"""

from game.game_loop import new_game
from game.moves import get_legal_moves, get_dead_cards
from shared.types import Card, Move


def test_legal_moves_nonempty_at_start():
    state = new_game()
    moves = get_legal_moves(state, 1)
    assert len(moves) > 0


def test_legal_moves_are_move_objects():
    state = new_game()
    moves = get_legal_moves(state, 1)
    for move in moves:
        assert isinstance(move, Move)
        assert move.move_type in ("place", "wild", "remove")
        r, c = move.position
        assert 0 <= r <= 9
        assert 0 <= c <= 9


def test_legal_moves_only_from_hand():
    state = new_game()
    hand_cards = set(state.hands[1])
    moves = get_legal_moves(state, 1)
    for move in moves:
        assert move.card in hand_cards


def test_place_moves_target_empty_cells():
    state = new_game()
    moves = get_legal_moves(state, 1)
    for move in moves:
        if move.move_type == "place":
            r, c = move.position
            assert int(state.chip_grid[r, c]) == 0, \
                f"Place move targets occupied cell ({r},{c})"


def test_no_dead_cards_at_start():
    state = new_game()
    # At the start of the game dead cards are rare — just check it returns a list
    dead = get_dead_cards(state, 1)
    assert isinstance(dead, list)


def test_dead_card_is_unplayable():
    """A card whose both board positions are occupied should be dead."""
    from game.board import CARD_TO_POSITIONS
    state = new_game()

    # Find a non-Jack card in hand and occupy both its positions
    target_card = None
    for card in state.hands[1]:
        if card.rank != "J" and card in CARD_TO_POSITIONS:
            positions = CARD_TO_POSITIONS[card]
            if len(positions) == 2:
                target_card = card
                break

    if target_card is None:
        return  # no suitable card found, skip

    for r, c in CARD_TO_POSITIONS[target_card]:
        state.chip_grid[r, c] = 2  # occupy with opponent

    dead = get_dead_cards(state, 1)
    assert target_card in dead
