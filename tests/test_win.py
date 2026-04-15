"""tests for game/win_detection.py"""

from game.game_loop import new_game
from game.win_detection import check_sequences, check_winner
from shared.types import SEQUENCES_TO_WIN


def test_no_winner_at_start():
    state = new_game()
    assert check_winner(state) == 0


def test_no_sequences_at_start():
    state = new_game()
    assert check_sequences(state, 1) == []
    assert check_sequences(state, 2) == []


def test_horizontal_sequence_detected():
    state = new_game()
    # Place 5 in a row horizontally for player 1 (row 1, cols 0-4)
    for c in range(5):
        state.chip_grid[1, c] = 1
    seqs = check_sequences(state, 1)
    assert len(seqs) >= 1


def test_vertical_sequence_detected():
    state = new_game()
    for r in range(5):
        state.chip_grid[r, 1] = 2
    seqs = check_sequences(state, 2)
    assert len(seqs) >= 1


def test_diagonal_sequence_detected():
    state = new_game()
    for i in range(5):
        state.chip_grid[i, i] = 1
    seqs = check_sequences(state, 1)
    assert len(seqs) >= 1


def test_corner_counts_as_wild():
    state = new_game()
    # (0,0) is a corner (-1), so placing 4 chips in the same line should form a sequence
    # Row 0: corner at col 0, place chips at cols 1-4
    for c in range(1, 5):
        state.chip_grid[0, c] = 1
    seqs = check_sequences(state, 1)
    assert len(seqs) >= 1


def test_winner_requires_two_sequences():
    state = new_game()
    required = SEQUENCES_TO_WIN[2]
    assert required == 2

    # One sequence should not win
    for c in range(5):
        state.chip_grid[1, c] = 1
    from game.win_detection import check_sequences as cs
    state.completed_sequences.update(set(cs(state, 1)[0]))
    state.sequence_counts[1] = 1
    assert check_winner(state) == 0

    # Second sequence wins
    for c in range(5):
        state.chip_grid[2, c] = 1
    state.sequence_counts[1] = 2
    assert check_winner(state) == 1


def test_opponent_chip_blocks_sequence():
    state = new_game()
    for c in range(4):
        state.chip_grid[1, c] = 1
    state.chip_grid[1, 4] = 2  # opponent blocks
    seqs = check_sequences(state, 1)
    # No complete sequence should be counted along this row
    for seq in seqs:
        positions = set(seq)
        assert (1, 4) not in positions
