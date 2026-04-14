# evaluation/run_post_rl.py
import json
from agent.combined_agent import CombinedAgent
from agent.greedy_agent import GreedyAgent
from game.agents.random_agent import RandomAgent
from evaluation.tournament import run_full_evaluation, print_summary_table
import agent.heuristic as h

# Load learned weights
with open("training/learned_weights.json") as f:
    data = json.load(f)
    learned = data["weights"]

# Swap in learned weights
original = h.DEFAULT_WEIGHTS.copy()
for k, v in learned.items():
    h.DEFAULT_WEIGHTS[k] = v

agents = {
    "Random": RandomAgent(),
    "Greedy": GreedyAgent(),
    "Combined (RL-tuned)": CombinedAgent(n_samples=3, depth=2),
}

results = run_full_evaluation(agents, n_games=50)
print_summary_table(results)

# Restore
for k, v in original.items():
    h.DEFAULT_WEIGHTS[k] = v
