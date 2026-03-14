import sys
from shared.types import Card, GameState
from game.moves import get_legal_moves, get_dead_cards
from game.win_detection import check_sequences, check_winner
from game.board import CARD_TO_POSITIONS, ALL_LINES, POSITION_TO_CARD

def run_tests():
    print("Initializing GameState...")
    state = GameState()
    
    print("Testing card deal...")
    # Deal a regular card, a two-eyed Jack, and a one-eyed Jack
    state.hands[1] = [
        Card("5", "hearts"),
        Card("J", "diamonds"), # wild
        Card("J", "spades")    # remove
    ]
    
    print("Testing legal moves generation for empty board...")
    moves = get_legal_moves(state, player=1)
    
    types = set(m.move_type for m in moves)
    assert "place" in types
    assert "wild" in types
    assert "remove" not in types # No opponent chips to remove
    
    print("Placing opponent chips to test 'remove'...")
    state.set_chip(0, 5, 2)
    state.set_chip(3, 3, 2)
    moves2 = get_legal_moves(state, player=1)
    types2 = set(m.move_type for m in moves2)
    assert "remove" in types2
    print(f"Total legal moves generated: {len(moves2)}")
    
    print("Testing win detection...")
    assert check_winner(state) == 0
    
    # Give player 1 a winning horizontal sequence on row 1
    # Note: (1,0) to (1,4)
    state.set_chip(1, 0, 1)
    state.set_chip(1, 1, 1)
    state.set_chip(1, 2, 1)
    state.set_chip(1, 3, 1)
    state.set_chip(1, 4, 1)
    
    # And a vertical sequence on column 9
    state.set_chip(1, 9, 1)
    state.set_chip(2, 9, 1)
    state.set_chip(3, 9, 1)
    state.set_chip(4, 9, 1)
    state.set_chip(5, 9, 1)
    
    winner = check_winner(state)
    assert winner == 1, f"Expected winner 1, got {winner}"
    
    print("All runtime logic checks passed! ✅")

if __name__ == "__main__":
    run_tests()
