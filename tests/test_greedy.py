from shared.types import Card, GameState
from game.game_loop import new_game, play_game
from game.moves import get_legal_moves
from game.agents.random_agent import RandomAgent
from agent.greedy_agent import GreedyAgent


def test_greedy_returns_legal_move():
    """Greedy agent must always return a legal move."""
    state = new_game()
    agent = GreedyAgent()
    move = agent.choose_move(state)
    legal = get_legal_moves(state, state.current_player)
    assert move in legal


def test_greedy_completes_game():
    """Greedy vs random should finish without crashing."""
    winner = play_game(GreedyAgent(), RandomAgent(), max_turns=300)
    assert winner in (0, 1, 2)


def test_greedy_beats_random():
    """Greedy should win more than 50% against random."""
    from game.game_loop import run_tournament
    results = run_tournament(GreedyAgent(), RandomAgent(), n_games=30)
    assert results[1] > results[2], "Greedy should beat random"


def test_greedy_prefers_winning_move():
    """Given a move that completes a sequence, greedy should pick it."""
    state = new_game()
    # Set up 4-in-a-row for player 1
    state.set_chip(0, 1, 1)  # 2 of spades position
    state.set_chip(0, 2, 1)  # 3 of spades position
    state.set_chip(0, 3, 1)  # 4 of spades position
    state.set_chip(0, 4, 1)  # 5 of spades position
    # Give player 1 a card that can complete at (0,5)
    from shared.types import Card
    state.hands[1] = [Card("6", "spades")]
    state.current_player = 1
    
    agent = GreedyAgent()
    move = agent.choose_move(state)
    # Should pick the completing move
    assert move.position == (0, 5) or move.card == Card("6", "spades")
