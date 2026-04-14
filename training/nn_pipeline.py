"""
training/nn_pipeline.py — Complete NN training pipeline.

Generates heuristic-labeled data, trains CNN, tests the agent.
Run with: python -m training.nn_pipeline
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from shared.types import GameState, next_player
from game.game_loop import new_game, apply_move, play_game
from game.moves import get_legal_moves, get_dead_cards
from game.agents.random_agent import RandomAgent
from agent.heuristic import evaluate
from agent.greedy_agent import GreedyAgent
from agent.combined_agent import CombinedAgent
from agent.nn_evaluator import (
    SequenceValueNet, encode_board, encode_hand, get_device
)


# ── Step 1: Generate Training Data ─────────────────────────────────

def generate_data(agent1, agent2, n_games, label=""):
    """Play games, record each board state with its heuristic score."""
    all_boards = []
    all_hands = []
    all_scores = []

    print(f"  Generating {n_games} {label} games...")

    for i in range(n_games):
        state = new_game()
        agents = {1: agent1, 2: agent2}

        for turn in range(300):
            player = state.current_player

            dead = get_dead_cards(state, player)
            for card in dead:
                state.hands[player].remove(card)
                state.discard_pile.append(card)
                if state.deck:
                    state.hands[player].append(state.deck.pop())

            moves = get_legal_moves(state, player)
            if not moves:
                state.current_player = next_player(player)
                continue

            # Record state labeled with heuristic score
            all_boards.append(encode_board(state, player))
            all_hands.append(encode_hand(state, player))
            all_scores.append(evaluate(state, player))

            move = agents[player].choose_move(state)
            winner = apply_move(state, move)
            if winner != 0:
                break

        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{n_games} games done, "
                  f"{len(all_scores)} samples")

    return all_boards, all_hands, all_scores


# ── Step 2: Train the CNN ───────────────────────────────────────────

def train_model(boards, hands, scores, save_path, epochs=80):
    """Train SequenceValueNet on heuristic-labeled data."""
    device = get_device()
    print(f"\n  Training on {device}...")

    boards_t = torch.stack(boards)
    hands_t = torch.stack(hands)
    scores_t = torch.tensor(scores, dtype=torch.float32)

    # Normalize scores
    score_mean = scores_t.mean().item()
    score_std = scores_t.std().item()
    if score_std > 0:
        scores_norm = (scores_t - score_mean) / score_std
    else:
        scores_norm = scores_t

    scores_norm = scores_norm.unsqueeze(1)

    # Split train/val
    n = len(boards_t)
    n_val = int(n * 0.1)
    n_train = n - n_val
    idx = torch.randperm(n)

    train_ds = TensorDataset(
        boards_t[idx[:n_train]], hands_t[idx[:n_train]],
        scores_norm[idx[:n_train]]
    )
    val_ds = TensorDataset(
        boards_t[idx[n_train:]], hands_t[idx[n_train:]],
        scores_norm[idx[n_train:]]
    )
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    print(f"  Samples: {n_train} train, {n_val} val")

    model = SequenceValueNet().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=8, factor=0.5
    )

    best_val = float('inf')

    for epoch in range(epochs):
        model.train()
        t_loss = 0.0
        for bb, bh, bs in train_loader:
            bb, bh, bs = bb.to(device), bh.to(device), bs.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bb, bh), bs)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * len(bb)
        t_loss /= n_train

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for bb, bh, bs in val_loader:
                bb, bh, bs = bb.to(device), bh.to(device), bs.to(device)
                v_loss += criterion(model(bb, bh), bs).item() * len(bb)
        v_loss /= n_val

        scheduler.step(v_loss)
        if v_loss < best_val:
            best_val = v_loss
            torch.save({
                "model_state": model.state_dict(),
                "score_mean": score_mean,
                "score_std": score_std,
            }, save_path)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            best_mark = " *best*" if v_loss <= best_val else ""
            print(f"    Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {t_loss:.6f} | Val: {v_loss:.6f} | "
                  f"LR: {lr:.6f}{best_mark}")

    print(f"  Best val loss: {best_val:.6f}")
    print(f"  Saved to {save_path}")


# ── Step 3: Test ────────────────────────────────────────────────────

def test_nn_agent(model_path):
    """Quick test: NN agent vs random."""
    from agent.nn_agent import NNAgent

    nn_agent = NNAgent(model_path=model_path, n_samples=3, depth=2)
    wins = 0
    for i in range(10):
        w = play_game(nn_agent, RandomAgent(), max_turns=300)
        if w == 1:
            wins += 1
        print(f"    Game {i+1}/10: {'WIN' if w == 1 else 'LOSS'}")
    print(f"  NN vs Random: {wins}/10 wins")
    return wins


# ── Main Pipeline ───────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("NN Training Pipeline")
    print("=" * 60)

    import os
    os.makedirs("training/models", exist_ok=True)

    model_path = "training/models/value_net_v2.pt"

    # Step 1: Generate data
    print("\n--- Step 1: Generate training data ---")
    start = time.time()

    greedy = GreedyAgent()
    combined = CombinedAgent(n_samples=2, depth=2)

    b1, h1, s1 = generate_data(greedy, greedy, 3000, "greedy")
    b2, h2, s2 = generate_data(combined, greedy, 2000, "combined")

    all_boards = b1 + b2
    all_hands = h1 + h2
    all_scores = s1 + s2

    gen_time = time.time() - start
    print(f"  Total: {len(all_scores)} samples in {gen_time/60:.1f} min")

    # Step 2: Train
    print("\n--- Step 2: Train CNN ---")
    start = time.time()
    train_model(all_boards, all_hands, all_scores, model_path, epochs=80)
    train_time = time.time() - start
    print(f"  Training time: {train_time/60:.1f} min")

    # Step 3: Test
    print("\n--- Step 3: Smoke test ---")
    test_nn_agent(model_path)

    print(f"\n{'='*60}")
    print(f"Total: {(gen_time + train_time)/60:.1f} min")
    print(f"Model: {model_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
