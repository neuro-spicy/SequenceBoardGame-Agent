"""
training/run_nn_training.py — Full NN training pipeline.

1. Generate training data from self-play games
2. Train the neural network
3. Verify the trained agent works
"""

import time
from game.agents.random_agent import RandomAgent
from agent.greedy_agent import GreedyAgent
from agent.combined_agent import CombinedAgent
from agent.nn_agent import NNAgent
from training.nn_data_generator import generate_dataset
from training.nn_trainer import train_value_network
from game.game_loop import play_game


def main():
    print("=" * 60)
    print("Neural Network Training Pipeline")
    print("=" * 60)
    
    # ── Step 1: Generate training data ──────────────────────
    print("\n--- Step 1: Generating training data ---\n")
    
    # Use greedy vs greedy for speed (first round)
    greedy = GreedyAgent()
    
    start = time.time()
    generate_dataset(
        greedy, greedy,
        n_games=5000,
        save_path="training/models/training_data.pt"
    )
    gen_time = time.time() - start
    print(f"Data generation time: {gen_time/60:.1f} minutes")
    
    # ── Step 2: Train the network ───────────────────────────
    print("\n--- Step 2: Training neural network ---\n")
    
    start = time.time()
    train_value_network(
        data_path="training/models/training_data.pt",
        save_path="training/models/value_net.pt",
        hidden_size=256,
        epochs=50,
        batch_size=256,
        learning_rate=0.001,
    )
    train_time = time.time() - start
    print(f"Training time: {train_time/60:.1f} minutes")
    
    # ── Step 3: Verify the trained agent ────────────────────
    print("\n--- Step 3: Smoke test ---\n")
    
    nn_agent = NNAgent(
        model_path="training/models/value_net.pt",
        n_samples=2, depth=2
    )
    
    # Quick test: 10 games vs random
    nn_wins = 0
    for i in range(10):
        winner = play_game(nn_agent, RandomAgent(), max_turns=300)
        if winner == 1:
            nn_wins += 1
    
    print(f"NN Agent vs Random: {nn_wins}/10 wins")
    
    if nn_wins >= 5:
        print("NN agent is working!")
    else:
        print("NN agent may need more training data or epochs.")
    
    # ── Optional Step 4: Iterative improvement ──────────────
    # Uncomment to generate better data using the NN agent itself
    # and retrain for stronger play:
    #
    # print("\n--- Step 4: Iterative improvement ---\n")
    # generate_dataset(
    #     nn_agent, nn_agent,
    #     n_games=2000,
    #     save_path="training/models/training_data_v2.pt"
    # )
    # train_value_network(
    #     data_path="training/models/training_data_v2.pt",
    #     save_path="training/models/value_net.pt",
    #     epochs=30,
    # )
    
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"Total time: {(gen_time + train_time)/60:.1f} minutes")
    print("Model saved to: training/models/value_net.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
