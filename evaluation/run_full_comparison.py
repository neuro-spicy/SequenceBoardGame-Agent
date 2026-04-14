"""
evaluation/run_full_comparison.py — All agents tournament.
"""

import os
import json
from game.agents.random_agent import RandomAgent
from agent.greedy_agent import GreedyAgent
from agent.combined_agent import CombinedAgent
from agent.nn_agent import NNAgent
from evaluation.tournament import run_full_evaluation, print_summary_table, save_results
import agent.heuristic as h


def main():
    print("=" * 60)
    print("Full Agent Comparison Tournament")
    print("=" * 60)

    # Load RL weights
    try:
        with open("training/learned_weights.json") as f:
            rl_data = json.load(f)
            rl_weights = rl_data["weights"]
        print(f"RL weights loaded: {rl_weights}")
    except FileNotFoundError:
        print("No RL weights found, skipping RL agent")
        rl_weights = None

    # Build agents
    agents = {
        "Random": RandomAgent(),
        "Greedy": GreedyAgent(),
        "Combined (hand-tuned)": CombinedAgent(n_samples=5, depth=3),
    }

    # Add RL-tuned agent if weights exist
    if rl_weights is not None:
        original_weights = h.DEFAULT_WEIGHTS.copy()
        for k, v in rl_weights.items():
            h.DEFAULT_WEIGHTS[k] = v
        agents["Combined (RL-tuned)"] = CombinedAgent(n_samples=5, depth=3)
        for k, v in original_weights.items():
            h.DEFAULT_WEIGHTS[k] = v

    # Add NN agent if model exists
    try:
        agents["NN Agent"] = NNAgent(
            model_path="training/models/value_net_v2.pt",
            n_samples=5, depth=3
        )
    except Exception as e:
        print(f"No NN model found, skipping NN agent: {e}")

    # Run all matchups
    print(f"\nRunning tournament with {len(agents)} agents...")
    results = run_full_evaluation(agents, n_games=50)
    print_summary_table(results)

    # Save results
    os.makedirs("evaluation/results", exist_ok=True)
    for r in results:
        fn = (f"final_{r['agent1']}_vs_{r['agent2']}.json"
              .replace(" ", "_").replace("(", "").replace(")", ""))
        save_results(r, f"evaluation/results/{fn}")


if __name__ == "__main__":
    main()
