"""
training/nn_trainer.py — Train the SequenceValueNet.

Improved with weight decay regularization and gradient clipping
for more stable training on larger datasets.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import TensorDataset, DataLoader
from agent.nn_evaluator import SequenceValueNet, get_device


def train_value_network(
    data_path: str = "training/models/training_data.pt",
    save_path: str = "training/models/value_net.pt",
    hidden_size: int = 256,
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    validation_split: float = 0.1,
):
    """
    Train the value network from saved training data.
    
    Parameters:
      data_path — path to the training data .pt file
      save_path — where to save the trained model
      hidden_size — neurons per hidden layer
      epochs — training epochs
      batch_size — mini-batch size
      learning_rate — Adam optimizer learning rate
      validation_split — fraction of data for validation
    """
    device = get_device()
    print(f"Training on device: {device}")
    
    # Load data
    data = torch.load(data_path)
    states = data["states"]
    labels = data["labels"].unsqueeze(1)  # shape: (N, 1)
    
    print(f"Dataset: {len(states)} samples, "
          f"input size: {states.shape[1]}")
    
    # Train/validation split
    n = len(states)
    n_val = int(n * validation_split)
    n_train = n - n_val
    
    indices = torch.randperm(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    train_dataset = TensorDataset(states[train_idx], labels[train_idx])
    val_dataset = TensorDataset(states[val_idx], labels[val_idx])
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size
    )
    
    print(f"Train: {n_train} samples | Val: {n_val} samples")
    
    # Create model
    model = SequenceValueNet(
        input_size=states.shape[1], 
        hidden_size=hidden_size
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    
    # Training setup — added weight decay for regularization
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 15  # stop if no improvement for 15 epochs
    
    # Training loop
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch_states, batch_labels in train_loader:
            batch_states = batch_states.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_states)
            loss = criterion(predictions, batch_labels)
            loss.backward()
            
            # Gradient clipping for stable training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * len(batch_states)
        
        train_loss /= n_train
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_states, batch_labels in val_loader:
                batch_states = batch_states.to(device)
                batch_labels = batch_labels.to(device)
                
                predictions = model(batch_states)
                loss = criterion(predictions, batch_labels)
                val_loss += loss.item() * len(batch_states)
        
        val_loss /= n_val
        scheduler.step(val_loss)
        
        # Save best model and track early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {train_loss:.4f} | "
                  f"Val: {val_loss:.4f} | "
                  f"LR: {lr:.6f} | "
                  f"{'*best*' if val_loss <= best_val_loss else ''}")
        
        # Early stopping — no point training if val loss plateaus
        if patience_counter >= early_stop_patience:
            print(f"\n  Early stopping at epoch {epoch+1} "
                  f"(no improvement for {early_stop_patience} epochs)")
            break
    
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to {save_path}")
    
    return model
