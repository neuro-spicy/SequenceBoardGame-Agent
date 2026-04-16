"""
training/nn_pipeline.py — continuous NN training pipeline.

alternates between generating self-play data and retraining the CNN.
saves the model after every training round, so Ctrl+C is safe.

run: python -m training.nn_pipeline
"""

import time
import os
import signal
import sys
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


MODEL_PATH = "training/models/value_net_v2.pt"


def generate_data(agent1, agent2, n_games, label=""):
    """play games, record each board state with its heuristic score."""
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

            # record state labeled with heuristic score
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


def train_model(boards, hands, scores, save_path, epochs=80, resume=False):
    """train SequenceValueNet on heuristic-labeled data."""
    device = get_device()
    print(f"\n  Training on {device}...")

    boards_t = torch.stack(boards)
    hands_t = torch.stack(hands)
    scores_t = torch.tensor(scores, dtype=torch.float32)

    # normalize scores
    score_mean = scores_t.mean().item()
    score_std = scores_t.std().item()
    if score_std > 0:
        scores_norm = (scores_t - score_mean) / score_std
    else:
        scores_norm = scores_t

    scores_norm = scores_norm.unsqueeze(1)

    # split train/val
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

    # try to load existing model for fine-tuning
    model = SequenceValueNet().to(device)
    if os.path.exists(save_path):
        checkpoint = torch.load(save_path, map_location=device)
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
            print(f"  Loaded existing model for fine-tuning")
        else:
            model.load_state_dict(checkpoint)
            print(f"  Loaded existing model for fine-tuning")

    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")

    criterion = nn.MSELoss()
    lr = 0.0005 if resume else 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)
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
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
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


def test_nn_agent(model_path):
    """quick test: NN agent vs greedy."""
    from agent.nn_agent import NNAgent

    nn_agent = NNAgent(model_path=model_path, n_samples=3, depth=2)
    greedy = GreedyAgent()
    nn_wins = 0
    for i in range(10):
        w = play_game(nn_agent, greedy, max_turns=300)
        if w == 1:
            nn_wins += 1
        print(f"    Game {i+1}/10: {'WIN' if w == 1 else 'LOSS'}")
    print(f"  NN vs Greedy: {nn_wins}/10 wins")
    return nn_wins

def main():
    print("=" * 60)
    print("Continuous NN Training Pipeline")
    print("=" * 60)
    print()
    print("Press Ctrl+C at any time to stop.")
    print("Model is saved after every training round.\n")

    os.makedirs("training/models", exist_ok=True)

    # handle Ctrl+C gracefully
    def handle_interrupt(sig, frame):
        print(f"\n\nTraining interrupted after round {round_num}.")
        print(f"Best model saved to {MODEL_PATH}")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)

    round_num = 0
    greedy = GreedyAgent()

    # round 1: initial training from scratch
    round_num = 1
    print(f"\n{'='*60}")
    print(f"Round {round_num}: Initial training (greedy + combined data)")
    print(f"{'='*60}")

    start = time.time()
    combined = CombinedAgent(n_samples=2, depth=2)

    data_save_path = "training/models/round1_data.pt"
    if os.path.exists(data_save_path):
        print(f"  Loading Round 1 data from {data_save_path}")
        data = torch.load(data_save_path)
        all_boards, all_hands, all_scores = data["boards"], data["hands"], data["scores"]
    else:
        b1, h1, s1 = generate_data(greedy, greedy, 3000, "greedy")
        b2, h2, s2 = generate_data(combined, greedy, 2000, "combined")

        all_boards = b1 + b2
        all_hands = h1 + h2
        all_scores = s1 + s2
        
        torch.save({
            "boards": all_boards,
            "hands": all_hands,
            "scores": all_scores
        }, data_save_path)

    print(f"  Total: {len(all_scores)} samples")
    train_model(all_boards, all_hands, all_scores, MODEL_PATH, epochs=80, resume=os.path.exists(MODEL_PATH))

    elapsed = time.time() - start
    print(f"  Round {round_num} complete in {elapsed/60:.1f} min")

    # quick test
    print(f"\n--- Smoke test after round {round_num} ---")
    test_nn_agent(MODEL_PATH)

    # rounds 2+: continuous self-play improvement
    while True:
        round_num += 1
        print(f"\n{'='*60}")
        print(f"Round {round_num}: Self-play improvement")
        print(f"{'='*60}")

        start = time.time()

        # load the current best model as an agent
        from agent.nn_agent import NNAgent
        nn_agent = NNAgent(model_path=MODEL_PATH, n_samples=2, depth=2)

        # generate new data: NN plays itself and vs greedy
        b1, h1, s1 = generate_data(nn_agent, nn_agent, 1500, "NN self-play")
        b2, h2, s2 = generate_data(nn_agent, greedy, 1000, "NN vs greedy")
        b3, h3, s3 = generate_data(combined, greedy, 500, "combined vs greedy")

        all_boards = b1 + b2 + b3
        all_hands = h1 + h2 + h3
        all_scores = s1 + s2 + s3

        print(f"  Total: {len(all_scores)} new samples")

        # fine-tune the existing model (it loads the checkpoint)
        train_model(all_boards, all_hands, all_scores, MODEL_PATH, epochs=40, resume=True)

        elapsed = time.time() - start
        print(f"  Round {round_num} complete in {elapsed/60:.1f} min")

        # test every round
        print(f"\n--- Test after round {round_num} ---")
        wins = test_nn_agent(MODEL_PATH)
        print(f"  Win rate vs Greedy: {wins}/10")


if __name__ == "__main__":
    main()
