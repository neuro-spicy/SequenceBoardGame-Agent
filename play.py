"""
play.py — interactive Sequence game (terminal).

modes:
  python play.py human-vs-ai
  python play.py human-vs-human
  python play.py ai-vs-ai
"""

import sys

from shared.types import BOARD_SIZE, CORNERS, CORNER_CHIP, next_player
from game.game_loop import new_game, apply_move, handle_dead_cards
from game.moves import get_legal_moves
from agent.combined_agent import CombinedAgent
from agent.greedy_agent import GreedyAgent


# board display symbols
_SYMBOLS = {
    CORNER_CHIP: " * ",  # corner (wild)
    0: " . ",             # empty
    1: " X ",             # player 1
    2: " O ",             # player 2
}

# completed-sequence positions shown with bracketed symbol
_SEQ_SYMBOLS = {1: "[X]", 2: "[O]"}


def _print_board(state):
    print()
    print("     " + "".join(f"{c:^4}" for c in range(BOARD_SIZE)))
    print("    " + "----" * BOARD_SIZE)
    for r in range(BOARD_SIZE):
        row_str = f" {r:2d} |"
        for c in range(BOARD_SIZE):
            chip = int(state.chip_grid[r, c])
            pos = (r, c)
            if pos in state.completed_sequences and chip in _SEQ_SYMBOLS:
                row_str += _SEQ_SYMBOLS[chip]
            else:
                row_str += _SYMBOLS.get(chip, " ? ")
            row_str += " " if c < BOARD_SIZE - 1 else ""
        print(row_str)
    print()
    print("  *=corner  X=Player1  O=Player2  .=empty  [X]/[O]=sequence")
    print()


def _print_hand(state, player):
    hand = state.hands[player]
    print(f"Your hand (Player {player}):")
    for i, card in enumerate(hand):
        print(f"  [{i}] {card}")


def _human_choose_move(state):
    player = state.current_player
    moves = get_legal_moves(state, player)

    _print_board(state)
    _print_hand(state, player)

    print(f"\nLegal moves ({len(moves)} available):")
    for i, move in enumerate(moves):
        r, c = move.position
        print(f"  [{i:2d}]  {move.card}  ->  ({r},{c})  [{move.move_type}]")

    while True:
        try:
            raw = input(f"\nPick move index (0-{len(moves)-1}), or 'q' to quit: ").strip()
            if raw.lower() == "q":
                print("Quitting.")
                sys.exit(0)
            choice = int(raw)
            if 0 <= choice < len(moves):
                return moves[choice]
            print(f"  Invalid — enter a number between 0 and {len(moves)-1}.")
        except ValueError:
            print("  Enter a number.")


def play_interactive(mode="human-vs-ai"):
    state = new_game()

    if mode == "human-vs-ai":
        ai = CombinedAgent(n_samples=5, depth=3)
        print("\n=== Sequence: Human vs AI ===")
        print("You are Player 1 (X). AI is Player 2 (O).\n")
    elif mode == "human-vs-human":
        print("\n=== Sequence: Human vs Human ===\n")
    elif mode == "ai-vs-ai":
        ai1 = CombinedAgent(n_samples=5, depth=3)
        ai2 = GreedyAgent()
        print("\n=== Sequence: Combined Agent (X) vs Greedy Agent (O) ===\n")

    for turn in range(500):
        player = state.current_player
        handle_dead_cards(state, player)
        moves = get_legal_moves(state, player)

        if not moves:
            state.current_player = next_player(player)
            continue

        if mode == "human-vs-ai":
            if player == 1:
                move = _human_choose_move(state)
            else:
                print("AI is thinking...")
                move = ai.choose_move(state)
                r, c = move.position
                print(f"AI plays {move.card} at ({r},{c}) [{move.move_type}]")

        elif mode == "human-vs-human":
            print(f"\n--- Player {player}'s turn ---")
            move = _human_choose_move(state)

        elif mode == "ai-vs-ai":
            if player == 1:
                move = ai1.choose_move(state)
            else:
                move = ai2.choose_move(state)
            r, c = move.position
            agent_name = "Combined" if player == 1 else "Greedy"
            print(f"Turn {turn+1:3d} | Player {player} ({agent_name}): "
                  f"{move.card}  ->  ({r},{c})  [{move.move_type}]")

        winner = apply_move(state, move)

        if winner != 0:
            _print_board(state)
            print("=" * 40)
            if mode == "human-vs-ai":
                label = "You win!" if winner == 1 else "AI wins!"
            else:
                label = f"Player {winner} wins!"
            print(f"  {label}")
            print("=" * 40)
            return winner

    print("Draw — maximum turns reached.")
    return 0


if __name__ == "__main__":
    valid = ["human-vs-ai", "human-vs-human", "ai-vs-ai"]
    mode = sys.argv[1] if len(sys.argv) > 1 else "human-vs-ai"
    if mode not in valid:
        print(f"Usage: python play.py [{' | '.join(valid)}]")
        sys.exit(1)
    play_interactive(mode)
