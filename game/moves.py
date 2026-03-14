"""
Functions for legal moves for a player and to identify dead cards.
"""

from typing import Optional

from shared.types import (
    Card, Move, GameState, BOARD_SIZE,
    is_one_eyed_jack, is_two_eyed_jack, is_jack,
    get_opponents,
)
from game.board import CARD_TO_POSITIONS


def get_legal_moves(
    state: GameState, player: Optional[int] = None
) -> list[Move]:
    """
    Return every legal move the given player can make.
    If player is None, defaults to state.current_player.
    Returns a list of Move - card, position and move_type.
    """
    if player is None:
        player = state.current_player

    hand = state.hands.get(player, [])
    moves: list[Move] = []
    seen_cards: set = set()

    for card in hand:
        if card in seen_cards:
            continue
        seen_cards.add(card)

        if is_two_eyed_jack(card):
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if state.chip_grid[r][c] == 0:
                        moves.append(Move(card, (r, c), "wild"))

        elif is_one_eyed_jack(card):
            opponents = get_opponents(player)
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if (
                        state.chip_grid[r][c] in opponents
                        and (r, c) not in state.completed_sequences
                    ):
                        moves.append(Move(card, (r, c), "remove"))

        else:
            positions = CARD_TO_POSITIONS.get(card, [])
            for pos in positions:
                r, c = pos
                if state.chip_grid[r][c] == 0:
                    moves.append(Move(card, pos, "place"))

    return moves


def get_dead_cards(
    state: GameState, player: Optional[int] = None
) -> list[Card]:
    """
    Return cards in the player's hand that cannot be played anywhere.
    A card is "dead" when every board position it maps to is already occupied.
    Jacks are never dead (they always have potential targets).
    If player is None, defaults to state.current_player.
    Returns a list of dead Cards.
    """
    if player is None:
        player = state.current_player

    hand = state.hands.get(player, [])
    dead: list[Card] = []

    for card in hand:
        if is_jack(card):
            continue

        positions = CARD_TO_POSITIONS.get(card, [])
        if not positions:
            dead.append(card)
            continue

        all_occupied = all(
            state.chip_grid[r][c] != 0 for r, c in positions
        )
        if all_occupied:
            dead.append(card)

    return dead
