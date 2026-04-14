"""
training/nn_data_generator.py — Generate training data via self-play.

Plays many games, records (state_encoding, outcome) pairs.
outcome = +1.0 if the current player at that state eventually won
outcome = -1.0 if they lost
outcome =  0.0 if draw

Includes dataset merging utility for combining multiple sources.
"""

import torch
import random
import os
from shared.types import GameState
from game.game_loop import new_game, apply_move
from game.moves import get_legal_moves, get_dead_cards
from shared.types import next_player
from agent.nn_evaluator import encode_state


def generate_game_data(agent1, agent2, max_turns=300):
    """
    Play one game and record all states with outcomes.
    
    Returns a list of (encoded_state, current_player, winner) tuples.
    We store current_player separately so we can compute the label 
    (did THIS player win?) after the game ends.
    """
    state = new_game()
    agents = {1: agent1, 2: agent2}
    
    # Record snapshots: (encoded_state_tensor, current_player)
    snapshots = []
    
    for turn in range(max_turns):
        player = state.current_player
        
        # Handle dead cards
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
        
        # Record the state BEFORE the move
        encoded = encode_state(state, player)
        snapshots.append((encoded, player))
        
        # Agent picks a move
        move = agents[player].choose_move(state)
        winner = apply_move(state, move)
        
        if winner != 0:
            # Game over — label all snapshots
            data = []
            for enc, snap_player in snapshots:
                if winner == snap_player:
                    label = 1.0
                else:
                    label = -1.0
                data.append((enc, label))
            return data
    
    # Draw — label everything 0
    return [(enc, 0.0) for enc, _ in snapshots]


def generate_dataset(
    agent1, agent2, 
    n_games: int = 5000,
    save_path: str = "training/models/training_data.pt"
) -> tuple:
    """
    Play n_games and collect all (state, outcome) training pairs.
    
    Parameters:
      agent1, agent2 — agents to play (can be the same for self-play)
      n_games — number of games to generate
      save_path — where to save the dataset
    
    Returns:
      (states_tensor, labels_tensor)
    """
    all_states = []
    all_labels = []
    
    print(f"Generating training data from {n_games} games...")
    
    for i in range(n_games):
        # Alternate who goes first
        if i % 2 == 0:
            data = generate_game_data(agent1, agent2)
        else:
            data = generate_game_data(agent2, agent1)
        
        for encoded_state, label in data:
            all_states.append(encoded_state)
            all_labels.append(label)
        
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{n_games} games | "
                  f"{len(all_states)} state-label pairs collected")
    
    states_tensor = torch.stack(all_states)
    labels_tensor = torch.tensor(all_labels, dtype=torch.float32)
    
    print(f"\nDataset: {len(all_states)} samples from {n_games} games")
    print(f"  Label distribution: "
          f"+1 (wins): {(labels_tensor == 1.0).sum().item()}, "
          f"-1 (losses): {(labels_tensor == -1.0).sum().item()}, "
          f"0 (draws): {(labels_tensor == 0.0).sum().item()}")
    
    # Save to disk
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"states": states_tensor, "labels": labels_tensor}, save_path)
    print(f"Saved to {save_path}")
    
    return states_tensor, labels_tensor


def merge_datasets(
    input_paths: list[str],
    output_path: str = "training/models/training_data_merged.pt"
) -> tuple:
    """
    Merge multiple .pt dataset files into one combined dataset.
    
    This lets us train on data from different agent matchups
    (combined vs combined, combined vs greedy, NN self-play, etc.)
    for a richer, more diverse training signal.
    
    Parameters:
      input_paths — list of paths to .pt files to merge
      output_path — where to save the merged dataset
    
    Returns:
      (states_tensor, labels_tensor)
    """
    all_states = []
    all_labels = []
    
    for path in input_paths:
        data = torch.load(path)
        all_states.append(data["states"])
        all_labels.append(data["labels"])
        print(f"  Loaded {len(data['states'])} samples from {path}")
    
    states_tensor = torch.cat(all_states)
    labels_tensor = torch.cat(all_labels)
    
    print(f"\nMerged dataset: {len(states_tensor)} total samples")
    print(f"  Label distribution: "
          f"+1 (wins): {(labels_tensor == 1.0).sum().item()}, "
          f"-1 (losses): {(labels_tensor == -1.0).sum().item()}, "
          f"0 (draws): {(labels_tensor == 0.0).sum().item()}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({"states": states_tensor, "labels": labels_tensor}, output_path)
    print(f"Saved merged dataset to {output_path}")
    
    return states_tensor, labels_tensor
