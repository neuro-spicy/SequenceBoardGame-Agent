# Phase 1 & 4
"""
Phase 4: Integration Test
"""

import time

from game.game_loop import play_game, run_tournament
from game.agents.random_agent import RandomAgent
from agent.combined_agent import CombinedAgent

def main():
    print("=" * 50)
    print("Phase 4: Integration Test")
    print("=" * 50)

    # 1. Smoke test
    print("\n--- Smoke Test (1 game, verbose) ---\n")
    ai = CombinedAgent(n_samples=2, depth=2)
    rand = RandomAgent()
    winner = play_game(ai, rand, verbose=True)
    print(f"\nWinner: Player {winner}")

    # 2. Performance check
    print("\n--- Performance Check ---\n")
    ai = CombinedAgent(n_samples=3, depth=2)
    state = None

    times = []
    from game.game_loop import new_game, apply_move
    from game.moves import get_legal_moves
    import random as rng

    state = new_game()
    for turn in range(20):
        player = state.current_player
        start = time.time()
        if player == 1:
            move = ai.choose_move(state)
        else:
            moves = get_legal_moves(state, player)
            move = rng.choice(moves) if moves else None

        if move:
            elapsed = time.time() - start
            if player == 1:
                times.append(elapsed)
            apply_move(state, move)

    if times:
        print(f"Avg AI decision time: {sum(times)/len(times):.2f}s")
        print(f"Max AI decision time: {max(times):.2f}s")

    # 3. Tournament
    print("\n--- Tournament: AI vs Random (50 games) ---\n")
    ai = CombinedAgent(n_samples=3, depth=2)
    results = run_tournament(ai, RandomAgent(), n_games=50)

    ai_win_rate = results[1] / 50.0
    print(f"\nAI win rate: {ai_win_rate:.0%}")

    if ai_win_rate > 0.6:
        print("Agent is performing well!")
    elif ai_win_rate > 0.5:
        print("Agent is slightly better than random. Consider tuning.")
    else:
        print("Agent is underperforming. Debug needed.")


if __name__ == "__main__":
    main()
