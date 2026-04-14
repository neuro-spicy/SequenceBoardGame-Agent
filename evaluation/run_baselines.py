# Phase 5
"""
Baseline tournament runner and results collection.
"""

from game.agents.random_agent import RandomAgent
from agent.combined_agent import CombinedAgent
from agent.greedy_agent import GreedyAgent
from evaluation.tournament import (
    run_full_evaluation, print_summary_table, save_results
)

def main():
    print("Running baseline tournaments (before RL)\n")
    
    agents = {
        "Random": RandomAgent(),
        "Greedy": GreedyAgent(),
        "Combined (hand-tuned)": CombinedAgent(n_samples=3, depth=2),
    }
    
    all_results = run_full_evaluation(agents, n_games=50)
    print_summary_table(all_results)
    
    # Save for comparison after RL
    for r in all_results:
        filename = (f"baseline_{r['agent1']}_vs_{r['agent2']}.json"
                   .replace(" ", "_").replace("(", "").replace(")", ""))
        save_results(r, filename)
    
    print("Baseline results saved. "
          "Run again after RL to compare.")

if __name__ == "__main__":
    main()
