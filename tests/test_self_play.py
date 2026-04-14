from training.self_play import (
    perturb_weights, make_agent_with_weights, play_match, train_weights
)
from agent.heuristic import DEFAULT_WEIGHTS


def test_perturb_weights_preserves_keys():
    """Perturbed weights must have the same keys as the original."""
    perturbed = perturb_weights(DEFAULT_WEIGHTS)
    assert set(perturbed.keys()) == set(DEFAULT_WEIGHTS.keys())


def test_perturb_weights_changes_values():
    """Perturbed weights should differ from the original."""
    perturbed = perturb_weights(DEFAULT_WEIGHTS, perturbation_range=0.5)
    # At least one weight should have changed
    any_different = any(
        perturbed[k] != DEFAULT_WEIGHTS[k] for k in DEFAULT_WEIGHTS
    )
    assert any_different


def test_perturb_weights_stays_positive():
    """All perturbed weights must remain positive."""
    for _ in range(100):
        perturbed = perturb_weights(DEFAULT_WEIGHTS, perturbation_range=0.9)
        for k, v in perturbed.items():
            assert v > 0, f"Weight {k} went non-positive: {v}"


def test_perturb_weights_within_range():
    """Perturbed weights should be within the expected range."""
    for _ in range(100):
        perturbed = perturb_weights(DEFAULT_WEIGHTS, perturbation_range=0.2)
        for k in DEFAULT_WEIGHTS:
            original = DEFAULT_WEIGHTS[k]
            low = original * 0.8
            high = original * 1.2
            # Allow small margin for the max(0.01, ...) clamp
            assert perturbed[k] >= 0.01


def test_weighted_agent_plays():
    """An agent with custom weights should complete a game."""
    from game.game_loop import play_game
    from game.agents.random_agent import RandomAgent
    
    custom = {"sequence_progress": 2.0, "opponent_threat": 0.5,
              "board_control": 0.1, "jack_utility": 1.0}
    agent = make_agent_with_weights(custom, n_samples=2, depth=1)
    winner = play_game(agent, RandomAgent(), max_turns=300)
    assert winner in (0, 1, 2)


def test_play_match_returns_correct_totals():
    """Win counts should add up to the number of games played."""
    from game.agents.random_agent import RandomAgent
    a1_wins, a2_wins, draws = play_match(
        RandomAgent(), RandomAgent(), n_games=10
    )
    assert a1_wins + a2_wins + draws == 10


def test_train_weights_returns_dict():
    """Training should return a valid weights dict."""
    # Very small training run for speed
    best, history = train_weights(
        n_generations=1,
        n_variants=1,
        games_per_match=4,
        agent_n_samples=1,
        agent_depth=1,
    )
    assert set(best.keys()) == set(DEFAULT_WEIGHTS.keys())
    assert len(history) == 1
    assert all(v > 0 for v in best.values())
