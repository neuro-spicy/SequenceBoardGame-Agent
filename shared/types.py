
"""
shared types and constants used by all modules.
"""

from typing import NamedTuple
import numpy as np


class Card(NamedTuple):
    """
    a single playing card.

    rank: "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"
    suit: "spades", "hearts", "diamonds", "clubs" are always in lowercase.
    """
    rank: str
    suit: str

    def __str__(self) -> str:
        return f"{self.rank} of {self.suit}"


def is_one_eyed_jack(card: Card) -> bool:
    """
    one-eyed jack (J of spades, J of hearts).
    used to REMOVE an opponent's chip.
    """
    return card.rank == "J" and card.suit in ("spades", "hearts")


def is_two_eyed_jack(card: Card) -> bool:
    """
    two-eyed jack (J of diamonds, J of clubs).
    wild — place your chip on any empty cell.
    """
    return card.rank == "J" and card.suit in ("diamonds", "clubs")


def is_jack(card: Card) -> bool:
    """True for any jack (one-eyed or two-eyed)."""
    return card.rank == "J"


class Move(NamedTuple):
    """
    a single player action.

    card:      the Card being played from hand
    position:  (row, col) on the board, 0-indexed
    move_type: exactly one of "place", "wild", "remove"
               "place"  - regular card, put your chip on the matching cell
               "wild"   - two-eyed jack, put your chip on any empty cell
               "remove" - one-eyed jack, remove one opponent chip
    """
    card: Card
    position: tuple[int, int]
    move_type: str          # "place" or "wild" or "remove"


BOARD_SIZE: int = 10

CORNERS: list[tuple[int, int]] = [
    (0, 0),
    (0, BOARD_SIZE - 1),
    (BOARD_SIZE - 1, 0),
    (BOARD_SIZE - 1, BOARD_SIZE - 1),
]

RANKS: list[str] = ["2", "3", "4", "5", "6", "7", "8", "9", "10",
                    "J", "Q", "K", "A"]

SUITS: list[str] = ["spades", "hearts", "diamonds", "clubs"]

HAND_SIZE: int = 7          # cards per player in a 2-player game

NUM_PLAYERS: int = 2

NUM_DECKS: int = 2          # two standard 52-card decks -> 104 cards

SEQUENCES_TO_WIN: dict[int, int] = {
    2: 2,   # 2 players/teams: need 2 sequences
    3: 1,   # 3 players/teams: need 1 sequence
}

SEQUENCE_LENGTH: int = 5    # 5 chips in a row to form a sequence


# chip_grid sentinel values
CORNER_CHIP: int = -1       # wild — counts for every player
EMPTY: int = 0
PLAYER_1: int = 1
PLAYER_2: int = 2


def get_opponents(player: int) -> list[int]:
    """return a list of opponent player numbers."""
    all_players = list(range(1, NUM_PLAYERS + 1))
    return [p for p in all_players if p != player]


def next_player(player: int) -> int:
    """return the player number whose turn comes next."""
    return (player % NUM_PLAYERS) + 1


class GameState:
    """
    complete snapshot of a Sequence game at one moment in time.

    attributes:
    chip_grid : np.ndarray (10x10, int32)
        values:
            -1  corner — wild, counts for all players
             0  empty
             1  player 1's chip
             2  player 2's chip

    hands : dict[int, list[Card]]
        {player_number: [list of Cards]}
        example: state.hands[1] -> player 1's current cards.

    deck : list[Card]
        draw pile. cards are drawn from the end (deck.pop()).

    discard_pile : list[Card]
        cards that have been played or declared dead.

    current_player : int
        whose turn it is (1 or 2).

    completed_sequences : set[tuple[int, int]]
        (row, col) positions that are part of a finished sequence.
        one-eyed jacks CANNOT remove chips from these positions.

    sequence_counts : dict[int, int]
        number of non-overlapping sequences each player has completed.
    """

    def __init__(self) -> None:
        # 10x10 board — corners set to -1, everything else 0
        self.chip_grid = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
        for r, c in CORNERS:
            self.chip_grid[r, c] = CORNER_CHIP

        self.hands: dict[int, list[Card]] = {}
        self.deck: list[Card] = []
        self.discard_pile: list[Card] = []
        self.current_player: int = PLAYER_1
        self.completed_sequences: set[tuple[int, int]] = set()
        self.sequence_counts: dict[int, int] = {1: 0, 2: 0}

    def get_chip(self, row: int, col: int) -> int:
        """return the chip value at (row, col)."""
        return int(self.chip_grid[row, col])

    def set_chip(self, row: int, col: int, player: int) -> None:
        """place player's chip at (row, col)."""
        self.chip_grid[row, col] = player

    def remove_chip(self, row: int, col: int) -> None:
        """remove the chip at (row, col), setting it back to EMPTY."""
        self.chip_grid[row, col] = EMPTY

    def copy(self) -> "GameState":
        """
        return a fast copy of the game state.

        lists and dicts are duplicated to avoid accidental sharing.
        cards and ints are shared safely because they are immutable.
        """
        new = object.__new__(GameState)
        new.chip_grid = self.chip_grid.copy()
        new.hands = {p: list(cards) for p, cards in self.hands.items()}
        new.deck = list(self.deck)
        new.discard_pile = list(self.discard_pile)
        new.current_player = self.current_player
        new.completed_sequences = set(self.completed_sequences)
        new.sequence_counts = dict(self.sequence_counts)
        return new

    def __repr__(self) -> str:
        return (
            f"GameState(current_player={self.current_player}, "
            f"deck_remaining={len(self.deck)}, "
            f"completed_sequences={len(self.completed_sequences)})"
        )
