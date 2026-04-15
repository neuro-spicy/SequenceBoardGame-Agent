"""
training/run_training.py — continuous RL weight training.

runs indefinitely, saving the best weights after every generation.
press Ctrl+C at any time to stop — the latest best weights are
already saved to training/learned_weights.json.
"""

import json
import signal
import sys
from training.self_play import train_weights, save_learned_weights
from agent.heuristic import DEFAULT_WEIGHTS


def main():
    print("=" * 50)
    print("Continuous Self-Play RL Weight Training")
    print("=" * 50)
    print()
    print("Press Ctrl+C at any time to stop.")
    print("Weights are saved after every generation.\n")

    # load previous best weights instead of starting from defaults
    try:
        with open("training/learned_weights.json") as f:
            data = json.load(f)
        best_weights = data["weights"]
        all_history = data.get("history", [])
        # resume the generation count from saved history
        generation = len(all_history)
        print(f"Resuming from generation {generation} with weights: {best_weights}")
    except (FileNotFoundError, json.JSONDecodeError):
        best_weights = DEFAULT_WEIGHTS.copy()
        all_history = []
        generation = 0
        print("No previous weights found, starting fresh")

    print()

    # handle Ctrl+C gracefully
    def handle_interrupt(sig, frame):
        print(f"\n\nTraining interrupted after {generation} generations.")
        print(f"Best weights saved to training/learned_weights.json")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)

    # run forever — each iteration is one generation (1 gen, 6 variants, 50 games each)
    while True:
        generation += 1

        new_weights, history = train_weights(
            initial_weights=best_weights.copy(),
            n_generations=1,
            n_variants=6,
            games_per_match=50,
            perturbation_range=0.15,
            agent_n_samples=2,
            agent_depth=2,
        )

        # check if weights improved this generation
        if new_weights != best_weights:
            best_weights = new_weights.copy()
            print(f"  >>> Generation {generation}: NEW BEST WEIGHTS FOUND")
        else:
            print(f"  >>> Generation {generation}: no improvement")

        # save after every generation
        all_history.extend(history)
        save_learned_weights(best_weights, all_history)

        # print running summary
        print(f"\n  [Gen {generation}] current best:")
        for key in best_weights:
            old = DEFAULT_WEIGHTS[key]
            new = best_weights[key]
            change = ((new - old) / old) * 100
            direction = "^" if new > old else "v"
            print(f"    {key}: {old:.2f} -> {new:.2f} "
                  f"({direction} {abs(change):.0f}%)")
        print()


if __name__ == "__main__":
    main()
