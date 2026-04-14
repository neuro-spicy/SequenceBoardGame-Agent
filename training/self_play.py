"""
training/self_play.py — Self-play RL for heuristic weight tuning.

Uses hill-climbing: perturb weights, play tournaments, keep winners.
"""

import random
import json
import os
import time
from agent.heuristic import DEFAULT_WEIGHTS
from agent.combined_agent import CombinedAgent
from game.game_loop import play_game


def perturb_weights(
    weights: dict, 
    perturbation_range: float = 0.2
) -> dict:
    """
    Create a new weights dict by randomly nudging each weight.
    
    Each weight is multiplied by a random factor in the range
    [1 - perturbation_range, 1 + perturbation_range].
    
    Example with perturbation_range=0.2:
      weight 1.0 could become anything from 0.8 to 1.2
    
    Parameters:
      weights — current weights dict
      perturbation_range — how much to vary (0.2 = ±20%)
    
    Returns:
      New weights dict with perturbed values.
    """
    new_weights = {}
    for key, value in weights.items():
        factor = 1.0 + random.uniform(
            -perturbation_range, perturbation_range
        )
        # Keep weights positive (negative weights don't make sense)
        new_weights[key] = max(0.01, value * factor)
    return new_weights


def make_agent_with_weights(
    weights: dict, 
    n_samples: int = 2, 
    depth: int = 2
) -> CombinedAgent:
    """
    Create a CombinedAgent that uses specific weights.
    
    Uses the simplest approach: temporarily swap DEFAULT_WEIGHTS
    in the heuristic module. The agent captures the weights at 
    creation time through a closure.
    """
    import agent.heuristic as h
    
    class WeightedAgent:
        """Wrapper that swaps weights before each decision."""
        def __init__(self):
            self.weights = weights.copy()
            self.inner = CombinedAgent(
                n_samples=n_samples, depth=depth
            )
        
        def choose_move(self, state):
            # Swap weights temporarily
            original = h.DEFAULT_WEIGHTS.copy()
            for k, v in self.weights.items():
                h.DEFAULT_WEIGHTS[k] = v
            
            move = self.inner.choose_move(state)
            
            # Restore original
            for k, v in original.items():
                h.DEFAULT_WEIGHTS[k] = v
            
            return move
    
    return WeightedAgent()


def play_match(
    agent1, agent2, n_games: int = 20
) -> tuple[int, int, int]:
    """
    Play n_games between two agents, alternating who goes first.
    Returns (agent1_wins, agent2_wins, draws).
    """
    a1_wins = 0
    a2_wins = 0
    draws = 0
    
    for i in range(n_games):
        if i % 2 == 0:
            winner = play_game(agent1, agent2, max_turns=300)
            if winner == 1:
                a1_wins += 1
            elif winner == 2:
                a2_wins += 1
            else:
                draws += 1
        else:
            winner = play_game(agent2, agent1, max_turns=300)
            if winner == 1:
                a2_wins += 1
            elif winner == 2:
                a1_wins += 1
            else:
                draws += 1
    
    return a1_wins, a2_wins, draws


def train_weights(
    initial_weights: dict = None,
    n_generations: int = 10,
    n_variants: int = 4,
    games_per_match: int = 20,
    perturbation_range: float = 0.2,
    agent_n_samples: int = 2,
    agent_depth: int = 2,
) -> dict:
    """
    Hill-climbing self-play to find optimal heuristic weights.
    
    Algorithm:
      1. Start with initial weights (default: DEFAULT_WEIGHTS)
      2. For each generation:
         a. Create n_variants perturbed weight sets
         b. Play each variant against the current best
         c. If any variant wins more than 50%, it becomes the new best
      3. Return the best weights found
    
    Parameters:
      initial_weights    — starting point (None = DEFAULT_WEIGHTS)
      n_generations      — number of improvement cycles
      n_variants         — perturbed variants to test per generation
      games_per_match    — games played per variant evaluation
      perturbation_range — how much to perturb (0.2 = ±20%)
      agent_n_samples    — determinization samples (keep low for speed)
      agent_depth        — search depth (keep low for speed)
    
    Returns:
      The best weights dict found after all generations.
    """
    if initial_weights is None:
        initial_weights = DEFAULT_WEIGHTS.copy()
    
    best_weights = initial_weights.copy()
    history = []  # track progress across generations
    
    print(f"Starting RL training")
    print(f"  Generations: {n_generations}")
    print(f"  Variants per generation: {n_variants}")
    print(f"  Games per match: {games_per_match}")
    print(f"  Initial weights: {best_weights}")
    print()
    
    for gen in range(n_generations):
        gen_start = time.time()
        print(f"--- Generation {gen + 1}/{n_generations} ---")
        
        best_agent = make_agent_with_weights(
            best_weights, agent_n_samples, agent_depth
        )
        
        gen_improved = False
        
        for v in range(n_variants):
            # Create a perturbed variant
            variant_weights = perturb_weights(
                best_weights, perturbation_range
            )
            variant_agent = make_agent_with_weights(
                variant_weights, agent_n_samples, agent_depth
            )
            
            # Play the variant against current best
            v_wins, b_wins, draws = play_match(
                variant_agent, best_agent, games_per_match
            )
            
            win_rate = v_wins / games_per_match
            print(f"  Variant {v+1}: {v_wins}W/{b_wins}L/{draws}D "
                  f"(win rate: {win_rate:.0%}) "
                  f"weights={_format_weights(variant_weights)}")
            
            # If variant is better, adopt its weights
            if v_wins > b_wins:
                print(f"  >>> New best found!")
                best_weights = variant_weights.copy()
                best_agent = make_agent_with_weights(
                    best_weights, agent_n_samples, agent_depth
                )
                gen_improved = True
        
        gen_time = time.time() - gen_start
        
        history.append({
            "generation": gen + 1,
            "best_weights": best_weights.copy(),
            "improved": gen_improved,
            "time_seconds": round(gen_time, 1),
        })
        
        print(f"  Best weights: {_format_weights(best_weights)}")
        print(f"  Time: {gen_time:.1f}s")
        print()
    
    print(f"Training complete!")
    print(f"Final weights: {best_weights}")
    
    return best_weights, history


def _format_weights(w: dict) -> str:
    """Short string representation of weights."""
    return (f"prog={w['sequence_progress']:.2f} "
            f"threat={w['opponent_threat']:.2f} "
            f"ctrl={w['board_control']:.2f} "
            f"jack={w['jack_utility']:.2f}")


def save_learned_weights(weights: dict, history: list) -> None:
    """Save the learned weights and training history to disk."""
    output = {
        "weights": weights,
        "history": history,
    }
    path = os.path.join("training", "learned_weights.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {path}")
