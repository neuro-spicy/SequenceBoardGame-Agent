"""
training/run_nn_training.py — Full NN training pipeline.

Improved pipeline with 3 phases:
  1. Generate high-quality data from combined agent self-play
  2. Train the neural network on that data
  3. Iterative improvement: NN plays itself, generates better data,
     retrains for even stronger play

This produces a model strong enough to beat the greedy agent.
"""

import time
import os
from game.agents.random_agent import RandomAgent
from agent.greedy_agent import GreedyAgent
from agent.combined_agent import CombinedAgent
from agent.nn_agent import NNAgent
from training.nn_data_generator import generate_dataset, merge_datasets
from training.nn_trainer import train_value_network
from game.game_loop import play_game


def main():
    print("=" * 60)
    print("Neural Network Training Pipeline (Improved)")
    print("=" * 60)
    
    os.makedirs("training/models", exist_ok=True)
    
    # ── Step 1: Generate high-quality training data ─────────
    # Use combined agent (minimax + heuristic) for strategic games.
    # These games contain real tactical decisions, not random noise.
    print("\n--- Step 1: Generating training data (combined agent) ---\n")
    
    # Use fast settings for data generation speed
    combined = CombinedAgent(n_samples=2, depth=2)
    greedy = GreedyAgent()
    
    start = time.time()
    
    # Phase 1a: combined vs combined — highest quality strategic play
    generate_dataset(
        combined, combined,
        n_games=2000,
        save_path="training/models/training_data_combined.pt"
    )
    
    # Phase 1b: combined vs greedy — learn how to beat greedy
    generate_dataset(
        combined, greedy,
        n_games=2000,
        save_path="training/models/training_data_vs_greedy.pt"
    )
    
    # Phase 1c: greedy vs greedy — fast baseline variety
    generate_dataset(
        greedy, greedy,
        n_games=3000,
        save_path="training/models/training_data_greedy.pt"
    )
    
    # Merge all datasets into one big training set
    merge_datasets(
        input_paths=[
            "training/models/training_data_combined.pt",
            "training/models/training_data_vs_greedy.pt",
            "training/models/training_data_greedy.pt",
        ],
        output_path="training/models/training_data.pt"
    )
    
    gen_time = time.time() - start
    print(f"\nData generation time: {gen_time/60:.1f} minutes")
    
    # ── Step 2: Train the network ───────────────────────────
    print("\n--- Step 2: Training neural network (round 1) ---\n")
    
    start = time.time()
    train_value_network(
        data_path="training/models/training_data.pt",
        save_path="training/models/value_net.pt",
        hidden_size=256,
        epochs=80,
        batch_size=512,
        learning_rate=0.001,
    )
    train_time = time.time() - start
    print(f"Training time: {train_time/60:.1f} minutes")
    
    # ── Step 3: Iterative improvement ───────────────────────
    # The NN agent plays itself, generating higher-quality data
    # that reflects its own strategic understanding. Then we
    # retrain on this improved data for a stronger model.
    print("\n--- Step 3: Iterative improvement (NN self-play) ---\n")
    
    nn_agent = NNAgent(
        model_path="training/models/value_net.pt",
        n_samples=2, depth=2
    )
    
    start_iter = time.time()
    
    # Generate games from NN self-play
    generate_dataset(
        nn_agent, nn_agent,
        n_games=2000,
        save_path="training/models/training_data_nn_selfplay.pt"
    )
    
    # Also play NN vs greedy to learn its weaknesses
    generate_dataset(
        nn_agent, greedy,
        n_games=1000,
        save_path="training/models/training_data_nn_vs_greedy.pt"
    )
    
    # Merge with original data for a richer dataset
    merge_datasets(
        input_paths=[
            "training/models/training_data.pt",
            "training/models/training_data_nn_selfplay.pt",
            "training/models/training_data_nn_vs_greedy.pt",
        ],
        output_path="training/models/training_data_v2.pt"
    )
    
    # Retrain on the combined dataset
    print("\n--- Step 3b: Retraining on improved data ---\n")
    train_value_network(
        data_path="training/models/training_data_v2.pt",
        save_path="training/models/value_net.pt",
        hidden_size=256,
        epochs=60,
        batch_size=512,
        learning_rate=0.0005,  # lower LR for fine-tuning
    )
    
    iter_time = time.time() - start_iter
    print(f"Iterative improvement time: {iter_time/60:.1f} minutes")
    
    # ── Step 4: Final smoke test ────────────────────────────
    print("\n--- Step 4: Smoke test ---\n")
    
    # Reload the improved model
    nn_agent = NNAgent(
        model_path="training/models/value_net.pt",
        n_samples=3, depth=2
    )
    
    # Test vs random
    nn_vs_random = 0
    for i in range(10):
        winner = play_game(nn_agent, RandomAgent(), max_turns=300)
        if winner == 1:
            nn_vs_random += 1
    print(f"NN Agent vs Random: {nn_vs_random}/10 wins")
    
    # Test vs greedy
    nn_vs_greedy = 0
    for i in range(10):
        winner = play_game(nn_agent, GreedyAgent(), max_turns=300)
        if winner == 1:
            nn_vs_greedy += 1
    print(f"NN Agent vs Greedy: {nn_vs_greedy}/10 wins")
    
    total_time = gen_time + train_time + iter_time
    
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Model saved to: training/models/value_net.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
