"""
game/game_loop.py — core game loop and simulation controller.
"""

from shared.types import GameState, HAND_SIZE, next_player, CORNER_CHIP
from game.deck import build_full_deck
from game.moves import get_legal_moves, get_dead_cards
from game.win_detection import check_sequences, check_winner


def new_game() -> GameState:
    """create a fresh game state ready to play."""
    state = GameState()
    state.deck = build_full_deck()
    state.hands = {1: [], 2: []}

    # deal 7 cards to each player, alternating
    for _ in range(HAND_SIZE):
        state.hands[1].append(state.deck.pop())
        state.hands[2].append(state.deck.pop())

    state.current_player = 1
    return state


def handle_dead_cards(state: GameState, player: int) -> None:
    """remove any dead cards and draw replacements before the player's turn."""
    dead_cards = get_dead_cards(state, player)
    for card in dead_cards:
        state.hands[player].remove(card)
        state.discard_pile.append(card)
        if state.deck:
            state.hands[player].append(state.deck.pop())


def apply_move(state: GameState, move) -> int:
    """
    execute a move on the given state (in place) and return the winner.
    returns 0 if the game is still ongoing.
    """
    player = state.current_player

    # remove the played card from the player's hand
    state.hands[player].remove(move.card)
    state.discard_pile.append(move.card)

    # execute on the board
    r, c = move.position
    if move.move_type in ("place", "wild"):
        state.set_chip(r, c, player)
    elif move.move_type == "remove":
        state.remove_chip(r, c)

    # check for new sequences only if a chip was placed
    if move.move_type in ("place", "wild"):
        for seq in check_sequences(state, player):
            # a sequence is new only if none of its non-corner positions
            # are already locked in a completed sequence
            non_corner = {pos for pos in seq
                          if state.chip_grid[pos[0], pos[1]] != CORNER_CHIP}
            if non_corner & state.completed_sequences:
                continue  # overlaps an existing sequence — skip
            for pos in seq:
                state.completed_sequences.add(pos)
            state.sequence_counts[player] += 1

    # check for winner
    winner = check_winner(state)

    # draw a new card if the deck is not empty
    if state.deck:
        state.hands[player].append(state.deck.pop())

    # switch to next player
    state.current_player = next_player(player)

    return winner


def play_game(agent1, agent2, verbose: bool = False, max_turns: int = 500) -> int:
    """run a complete game between two agents and return the winner (0 = draw)."""
    state = new_game()
    agents = {1: agent1, 2: agent2}

    for turn in range(max_turns):
        player = state.current_player
        agent = agents[player]

        handle_dead_cards(state, player)
        moves = get_legal_moves(state, player)

        if not moves:
            # no moves available — skip this player's turn
            state.current_player = next_player(player)
            continue

        move = agent.choose_move(state)
        if verbose:
            print(f"Turn {turn + 1} | Player {player} plays {move.card} "
                  f"at {move.position} ({move.move_type})")

        winner = apply_move(state, move)
        if winner != 0:
            if verbose:
                print(f"Winner: Player {winner}!")
            return winner

    return 0  # draw


def run_tournament(agent1, agent2, n_games: int = 100) -> dict:
    """play n_games and report win/loss statistics."""
    results = {1: 0, 2: 0, 0: 0}

    for game in range(n_games):
        if game % 2 == 0:
            winner = play_game(agent1, agent2)
            results[winner] += 1
        else:
            winner = play_game(agent2, agent1)
            if winner == 1:
                results[2] += 1
            elif winner == 2:
                results[1] += 1
            else:
                results[0] += 1

    print(f"Tournament Results ({n_games} games):")
    print(f"Agent 1 Won {results[1]} ({(results[1]/n_games)*100:.1f}%)")
    print(f"Agent 2 Won {results[2]} ({(results[2]/n_games)*100:.1f}%)")
    print(f"Draws: {results[0]} ({(results[0]/n_games)*100:.1f}%)")

    return results
