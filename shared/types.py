"""
shared/types.py - Shared types and constants used by all modules.
"""

from typing import NamedTuple, List, Dict, Set, Tuple, Optional


# ── Card ────────────────────────────────────────────────────────────

class Card(NamedTuple):
    """
    A single playing card.

    rank: "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"
    suit: "spades", "hearts", "diamonds", "clubs" are always in lowercase.
    """
    rank: str
    suit: str

    def __str__(self) -> str:
        return f"{self.rank} of {self.suit}"


# ── Jack helpers ────────────────────────────────────────────────────

def is_one_eyed_jack(card: Card) -> bool:
    """One-eyed Jack (J of spades, J of hearts) - used to REMOVE an opponent's chip."""
    return card.rank == "J" and card.suit in ("spades", "hearts")


def is_two_eyed_jack(card: Card) -> bool:
    """Two-eyed Jack (J of diamonds, J of clubs) - wild, place your chip on ANY empty cell."""
    return card.rank == "J" and card.suit in ("diamonds", "clubs")


def is_jack(card: Card) -> bool:
    """True for any Jack (one-eyed or two-eyed)."""
    return card.rank == "J"


# ── Move ────────────────────────────────────────────────────────────

class Move(NamedTuple):
    """
    A single player action.

    card:      the Card being played from hand
    position:  (row, col) on the board, 0-indexed
    move_type: exactly one of "place", "wild", "remove"
               "place"  - regular card, put your chip on the matching cell
               "wild"   - two-eyed Jack, put your chip on any empty cell
               "remove" - one-eyed Jack, remove one opponent chip
    """
    card: Card
    position: Tuple[int, int]
    move_type: str          # "place" | "wild" | "remove"


# ── Game configuration constants ────────────────────────────────────

BOARD_SIZE: int = 10

CORNERS: List[Tuple[int, int]] = [
    (0, 0),
    (0, BOARD_SIZE - 1),
    (BOARD_SIZE - 1, 0),
    (BOARD_SIZE - 1, BOARD_SIZE - 1),
]

RANKS: List[str] = ["2", "3", "4", "5", "6", "7", "8", "9", "10",
                     "J", "Q", "K", "A"]

SUITS: List[str] = ["spades", "hearts", "diamonds", "clubs"]

HAND_SIZE: int = 7          # cards per player in a 2-player game

NUM_PLAYERS: int = 2

NUM_DECKS: int = 2          # two standard 52-card decks → 104 cards

SEQUENCES_TO_WIN: Dict[int, int] = {
    2: 2,   # 2 players/teams: need 2 sequences
    3: 1,   # 3 players/teams: need 1 sequence
}

SEQUENCE_LENGTH: int = 5    # five-in-a-row


# chip_grid sentinel values
CORNER_CHIP: int = -1       # wild - counts for every player
EMPTY: int = 0
PLAYER_1: int = 1
PLAYER_2: int = 2


# ── Player helpers ──────────────────────────────────────────────────

def get_opponents(player: int) -> List[int]:
    """
    Return a list of opponent player numbers.
    """
    all_players = list(range(1, NUM_PLAYERS + 1))
    return [p for p in all_players if p != player]


def next_player(player: int) -> int:
    """Return the player number whose turn comes next."""
    return (player % NUM_PLAYERS) + 1


# ── GameState ───────────────────────────────────────────────────────

class GameState:
    """
    Complete snapshot of a Sequence game at one moment in time.

    Attributes
    ----------
    chip_grid : List[List[int]]
        10×10 grid.  Values:
            -1  corner (wild, counts for all players)
             0  empty
             1  player 1's chip
             2  player 2's chip

    hands : Dict[int, List[Card]]
        {player_number: [list of Cards]}.
        Example: state.hands[1] → player 1's current cards.

    deck : List[Card]
        Draw pile. Cards are drawn from the end (deck.pop()).

    discard_pile : List[Card]
        Cards that have been played or declared dead.

    current_player : int
        Whose turn it is (1 or 2).

    completed_sequences : Set[Tuple[int, int]]
        (row, col) positions that are part of a finished sequence.
        One-eyed Jacks CANNOT remove chips from these positions.
    """

    def __init__(self) -> None:
        # 10x10 board - corners set to -1, everything else 0
        self.chip_grid: List[List[int]] = [
            [EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)
        ]
        for r, c in CORNERS:
            self.chip_grid[r][c] = CORNER_CHIP

        self.hands: Dict[int, List[Card]] = {}
        self.deck: List[Card] = []
        self.discard_pile: List[Card] = []
        self.current_player: int = PLAYER_1
        self.completed_sequences: Set[Tuple[int, int]] = set()

    # ── grid helpers ────────────────────────────────────────────────

    def get_chip(self, row: int, col: int) -> int:
        """Return the chip value at (row, col)."""
        return self.chip_grid[row][col]

    def set_chip(self, row: int, col: int, player: int) -> None:
        """Place *player*'s chip at (row, col)."""
        self.chip_grid[row][col] = player

    def remove_chip(self, row: int, col: int) -> None:
        """Remove the chip at (row, col), setting it back to EMPTY."""
        self.chip_grid[row][col] = EMPTY

    # ── copying ─────────────────────────────────────────────────────

    def copy(self) -> "GameState":
        """
        Return a fast copy of the game state.

        Lists and dictionaries are duplicated to avoid accidental sharing.
        Cards and ints are shared safely because they cannot be changed.
        This is much faster than deepcopy, which is important for the AI.
        """
        new = GameState()
        new.chip_grid = [row[:] for row in self.chip_grid]
        new.hands = {p: list(cards) for p, cards in self.hands.items()}
        new.deck = list(self.deck)
        new.discard_pile = list(self.discard_pile)
        new.current_player = self.current_player
        new.completed_sequences = set(self.completed_sequences)
        return new

    # ── dunder helpers ──────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"GameState(current_player={self.current_player}, "
            f"deck_remaining={len(self.deck)}, "
            f"completed_sequences={len(self.completed_sequences)})"
        )
