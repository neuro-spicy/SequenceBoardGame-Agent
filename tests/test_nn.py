"""
tests/test_nn.py — Tests for the neural network components.
"""

import torch
from shared.types import GameState, Card, HAND_SIZE
from game.game_loop import new_game
from agent.nn_evaluator import (
    encode_board, encode_hand, encode_state,
    SequenceValueNet, CARD_INDEX, NUM_CARD_TYPES
)


def test_board_encoding_shape():
    """Board encoding must be 4 channels × 10 × 10."""
    state = new_game()
    encoded = encode_board(state, player=1)
    assert encoded.shape == (4, 10, 10)


def test_board_encoding_values():
    """Board encoding should only contain 0.0 and 1.0."""
    state = new_game()
    encoded = encode_board(state, player=1)
    unique = torch.unique(encoded)
    for v in unique:
        assert v.item() in (0.0, 1.0)


def test_hand_encoding_shape():
    """Hand encoding must be length 52."""
    state = new_game()
    encoded = encode_hand(state, player=1)
    assert encoded.shape == (NUM_CARD_TYPES,)


def test_hand_encoding_sum():
    """Hand encoding values should sum to hand size."""
    state = new_game()
    encoded = encode_hand(state, player=1)
    assert encoded.sum().item() == HAND_SIZE


def test_full_encoding_shape():
    """Full state encoding must be length 452 (400 + 52)."""
    state = new_game()
    encoded = encode_state(state, player=1)
    assert encoded.shape == (452,)


def test_network_forward_pass():
    """Network should accept encoded state and output one value."""
    model = SequenceValueNet(input_size=452, hidden_size=128)
    state = new_game()
    encoded = encode_state(state, player=1).unsqueeze(0)
    output = model(encoded)
    assert output.shape == (1, 1)


def test_network_output_range():
    """Output should be between -1 and +1 (tanh activation)."""
    model = SequenceValueNet(input_size=452, hidden_size=128)
    state = new_game()
    encoded = encode_state(state, player=1).unsqueeze(0)
    output = model(encoded).item()
    assert -1.0 <= output <= 1.0


def test_card_index_completeness():
    """Every rank-suit combo should have a unique index 0-51."""
    assert len(CARD_INDEX) == 52
    indices = set(CARD_INDEX.values())
    assert indices == set(range(52))


def test_encoding_perspective():
    """Encoding for player 1 and player 2 should be different
    when both have chips on the board."""
    state = new_game()
    state.set_chip(2, 3, 1)  # player 1 chip
    state.set_chip(4, 5, 2)  # player 2 chip
    
    enc1 = encode_board(state, player=1)
    enc2 = encode_board(state, player=2)
    
    # Channel 0 (our chips) should be different
    assert not torch.equal(enc1[0], enc2[0])
    # Channel 1 (opponent chips) should be different  
    assert not torch.equal(enc1[1], enc2[1])


def test_batch_forward():
    """Network should handle batched inputs."""
    model = SequenceValueNet(input_size=452, hidden_size=128)
    batch = torch.randn(16, 452)
    output = model(batch)
    assert output.shape == (16, 1)
