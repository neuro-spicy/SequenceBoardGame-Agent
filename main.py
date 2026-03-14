import sys
import random
from game.game_loop import play_game, run_tournament
from game.agents.random_agent import RandomAgent


def run_tests():
    agent1 = RandomAgent()
    agent2 = RandomAgent()

    # Single game to see every move
    print("Running a single verbose game...\n")
    winner = play_game(agent1, agent2, verbose=True)
    print(f"\nSingle game complete. Winner: {winner}\n")
    
    # 100 games
    print("Running a 100-game tournament to verify engine\n")
    results = run_tournament(agent1, agent2, n_games=100)
    print("\nAll game engine test scripts ran")


if __name__ == "__main__":
    run_tests()
