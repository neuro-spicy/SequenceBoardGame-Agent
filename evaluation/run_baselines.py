"""
evaluation/run_baselines.py — baseline tournament runner.

runs Random, Greedy, and Combined (hand-tuned) against each other.
"""

from game.agents.random_agent import RandomAgent
from agent.combined_agent import CombinedAgent
from agent.greedy_agent import GreedyAgent
from evaluation.tournament import (
    run_full_evaluation, print_summary_table, save_results
)


def main():
    print("Running baseline tournaments\n")

    agents = {
        "Random": RandomAgent(),
        "Greedy": GreedyAgent(),
        "Combined (hand-tuned)": CombinedAgent(n_samples=3, depth=2),
    }

    all_results = run_full_evaluation(agents, n_games=50)
    print_summary_table(all_results)

    # save for comparison after RL tuning
    for r in all_results:
        filename = (f"baseline_{r['agent1']}_vs_{r['agent2']}.json"
                   .replace(" ", "_").replace("(", "").replace(")", ""))
        save_results(r, filename)

    print("Baseline results saved.")


if __name__ == "__main__":
    main()
