# training/run_training.py

from training.self_play import train_weights, save_learned_weights
from agent.heuristic import DEFAULT_WEIGHTS


def main():
    print("=" * 50)
    print("Phase 5: Self-Play RL Weight Training")
    print("=" * 50)
    print()
    
    # Start from the hand-tuned defaults
    print(f"Starting weights: {DEFAULT_WEIGHTS}")
    print()
    
    # Training settings
    # Use low agent settings for speed during training
    # n_generations=10 with n_variants=4 = 40 matchups
    # Each matchup = 20 games
    # Total = ~800 games
    best_weights, history = train_weights(
        initial_weights=DEFAULT_WEIGHTS.copy(),
        n_generations=2,
        n_variants=2,
        games_per_match=2,
        perturbation_range=0.2,
        agent_n_samples=2,   # keep low for speed
        agent_depth=2,       # keep low for speed
    )
    
    # Save results
    save_learned_weights(best_weights, history)
    
    # Print comparison
    print("\n" + "=" * 50)
    print("Comparison:")
    print(f"  Hand-tuned: {DEFAULT_WEIGHTS}")
    print(f"  RL-learned: {best_weights}")
    print("=" * 50)
    
    # Show which weights changed most
    print("\nWeight changes:")
    for key in DEFAULT_WEIGHTS:
        old = DEFAULT_WEIGHTS[key]
        new = best_weights[key]
        change = ((new - old) / old) * 100
        direction = "↑" if new > old else "↓"
        print(f"  {key}: {old:.2f} → {new:.2f} "
              f"({direction} {abs(change):.0f}%)")


if __name__ == "__main__":
    main()
