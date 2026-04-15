"""
evaluation/run_ablation.py — ablation studies: measure each component's contribution.

compares three ablated agents against GreedyAgent as baseline:
  - SearchOnly:  minimax depth=3, no belief sampling
  - BeliefOnly:  determinization + greedy 1-ply, no deep search
  - CombinedAgent: full system (search + belief)

run: python -m evaluation.run_ablation
"""

from collections import defaultdict

from evaluation.tournament import run_matchup, print_matchup, save_results
from game.moves import get_legal_moves
from agent.greedy_agent import GreedyAgent
from agent.combined_agent import CombinedAgent
from agent.search import minimax_search, _apply_move_for_search
from agent.belief import policy_average_search
from agent.heuristic import evaluate


class SearchOnlyAgent:
    """minimax search with no belief model. searches the real (partially
    observable) state directly — opponent hand is whatever it happens to be."""

    def __init__(self, depth=3):
        self.depth = depth

    def choose_move(self, state):
        player = state.current_player
        move_scores = minimax_search(state, self.depth, player)
        if not move_scores:
            return None
        return max(move_scores, key=move_scores.get)


class BeliefOnlyAgent:
    """determinization + 1-ply greedy averaging. no deep search."""

    def __init__(self, n_samples=5):
        self.n_samples = n_samples

    def choose_move(self, state):
        player = state.current_player

        def greedy_search(s, depth, p):
            moves = get_legal_moves(s, p)
            scores = {}
            for move in moves:
                child = s.copy()
                _apply_move_for_search(child, move)
                scores[move] = evaluate(child, p)
            return scores

        avg = policy_average_search(
            state, player, greedy_search, self.n_samples, depth=1
        )
        if not avg:
            return None
        return max(avg, key=avg.get)


def main():
    print("=" * 60)
    print("Ablation Studies")
    print("Baseline: GreedyAgent (heuristic only, no search/belief)")
    print("=" * 60)

    matchups = [
        ("Search only (no belief)", SearchOnlyAgent(depth=3)),
        ("Belief only (no search)", BeliefOnlyAgent(n_samples=5)),
        ("Full combined", CombinedAgent(n_samples=5, depth=3)),
    ]

    all_results = []
    for name, agent in matchups:
        print(f"\n--- {name} vs Greedy ---")
        result = run_matchup(agent, GreedyAgent(), name, "Greedy", n_games=50)
        print_matchup(result)
        all_results.append(result)

    # summary table
    print("=" * 60)
    print(f"{'Agent':<30} {'vs Greedy Win%':>14}")
    print("-" * 60)
    print(f"{'Greedy (baseline)':<30} {'--':>14}")
    for r in all_results:
        print(f"{r['agent1']:<30} {r['agent1_win_rate']:>13.0%}")
    print("=" * 60)

    save_results(all_results, "ablation_results.json")


if __name__ == "__main__":
    main()
