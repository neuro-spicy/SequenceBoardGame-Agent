"""
evaluation/tournament.py — tournament runner and results collector.

runs any-vs-any agent tournaments and stores results in a
structured format for analysis and reporting.
"""

import time
import json
import os
from game.game_loop import play_game


def run_matchup(
    agent1, agent2,
    agent1_name: str,
    agent2_name: str,
    n_games: int = 50
) -> dict:
    """
    run a tournament between two agents and return detailed results.

    alternates who goes first for fairness.

    returns a dict with:
      - agent1_name, agent2_name
      - agent1_wins, agent2_wins, draws
      - agent1_win_rate
      - avg_game_length (turns)
      - total_time
    """
    results = {
        "agent1": agent1_name,
        "agent2": agent2_name,
        "n_games": n_games,
        "agent1_wins": 0,
        "agent2_wins": 0,
        "draws": 0,
        "game_lengths": [],
    }

    start_time = time.time()

    for i in range(n_games):
        if i % 2 == 0:
            winner = play_game(agent1, agent2, max_turns=500)
            if winner == 1:
                results["agent1_wins"] += 1
            elif winner == 2:
                results["agent2_wins"] += 1
            else:
                results["draws"] += 1
        else:
            winner = play_game(agent2, agent1, max_turns=500)
            if winner == 1:
                results["agent2_wins"] += 1
            elif winner == 2:
                results["agent1_wins"] += 1
            else:
                results["draws"] += 1

        if (i + 1) % 10 == 0:
            print(f"  {agent1_name} vs {agent2_name}: "
                  f"{i+1}/{n_games} games complete")

    elapsed = time.time() - start_time

    results["agent1_win_rate"] = results["agent1_wins"] / n_games
    results["agent2_win_rate"] = results["agent2_wins"] / n_games
    results["draw_rate"] = results["draws"] / n_games
    results["total_time_seconds"] = round(elapsed, 1)

    return results


def print_matchup(results: dict) -> None:
    """pretty-print a matchup result."""
    a1 = results["agent1"]
    a2 = results["agent2"]
    n = results["n_games"]

    print(f"\n{'='*50}")
    print(f"{a1} vs {a2} ({n} games)")
    print(f"{'='*50}")
    print(f"{a1}: {results['agent1_wins']} wins "
          f"({results['agent1_win_rate']:.0%})")
    print(f"{a2}: {results['agent2_wins']} wins "
          f"({results['agent2_win_rate']:.0%})")
    print(f"Draws: {results['draws']} "
          f"({results['draw_rate']:.0%})")
    print(f"Time: {results['total_time_seconds']}s")
    print(f"{'='*50}\n")


def save_results(results: dict, filename: str) -> None:
    """save results to a JSON file in evaluation/results/."""
    path = os.path.join("evaluation", "results", filename)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")


def run_full_evaluation(agents: dict, n_games: int = 50) -> list:
    """
    run every pair of agents against each other.

    parameters:
      agents — dict of {"name": agent_instance}
      n_games — games per matchup

    returns:
      list of matchup result dicts.
    """
    names = list(agents.keys())
    all_results = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name1, name2 = names[i], names[j]
            print(f"\nRunning: {name1} vs {name2}")

            result = run_matchup(
                agents[name1], agents[name2],
                name1, name2, n_games
            )
            print_matchup(result)
            all_results.append(result)

    return all_results


def print_summary_table(all_results: list) -> None:
    """print a summary table of all matchups."""
    print(f"\n{'='*60}")
    print(f"{'Matchup':<35} {'Win%':>6} {'Lose%':>6} {'Draw%':>6}")
    print(f"{'-'*60}")

    for r in all_results:
        matchup = f"{r['agent1']} vs {r['agent2']}"
        print(f"{matchup:<35} "
              f"{r['agent1_win_rate']:>5.0%} "
              f"{r['agent2_win_rate']:>5.0%} "
              f"{r['draw_rate']:>5.0%}")

    print(f"{'='*60}\n")
