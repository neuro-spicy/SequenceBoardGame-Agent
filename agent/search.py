"""
agent/search.py — minimax search with alpha-beta pruning.

uses _apply_move_for_search instead of game_loop.apply_move:
  - no random card draws during search (removes noise)
  - skips discard/sequence tracking for speed
"""

from shared.types import GameState, Move, next_player
from game.moves import get_legal_moves
from game.win_detection import check_winner
from agent.heuristic import evaluate, POSITION_VALUES

MAX_MOVES_TO_SEARCH = 15


def _apply_move_for_search(state, move):
    """
    lightweight move application for search only.
    no card draw, no discard pile, no sequence tracking.
    """
    player = state.current_player
    r, c = move.position

    # remove card from hand
    state.hands[player].remove(move.card)

    # execute on board
    if move.move_type in ("place", "wild"):
        state.chip_grid[r, c] = player
    elif move.move_type == "remove":
        state.chip_grid[r, c] = 0

    # switch player
    state.current_player = next_player(player)


def _order_moves(state: GameState, moves: list[Move], player: int) -> list[Move]:
    """sort moves by quick 1-ply evaluation of the resulting state."""
    def move_priority(move: Move):
        child = state.copy()
        _apply_move_for_search(child, move)
        return evaluate(child, player)

    return sorted(moves, key=move_priority, reverse=True)


def _minimax(
    state: GameState,
    depth: int,
    root_player: int,
    is_maximizing: bool,
    alpha: float,
    beta: float
) -> float:
    """recursive minimax with alpha-beta pruning."""
    winner = check_winner(state)
    if winner == root_player:
        return 10000.0 + depth
    elif winner != 0 and winner is not None:
        return -10000.0 - depth

    if depth == 0:
        return evaluate(state, root_player)

    moves = get_legal_moves(state, state.current_player)
    if not moves:
        return evaluate(state, root_player)

    ordered_moves = _order_moves(state, moves, state.current_player)[:MAX_MOVES_TO_SEARCH]

    if is_maximizing:
        max_eval = float('-inf')
        for move in ordered_moves:
            child = state.copy()
            _apply_move_for_search(child, move)
            eval_score = _minimax(
                child, depth - 1, root_player,
                False, alpha, beta
            )
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in ordered_moves:
            child = state.copy()
            _apply_move_for_search(child, move)
            eval_score = _minimax(
                child, depth - 1, root_player,
                True, alpha, beta
            )
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval


def minimax_search(state: GameState, depth: int, player: int) -> dict[Move, float]:
    """
    run minimax with alpha-beta pruning on a fully observable state.
    returns a dict mapping each legal Move to its minimax score.
    """
    original_player = state.current_player
    state.current_player = player

    moves = get_legal_moves(state, player)
    move_scores: dict[Move, float] = {}

    alpha = float('-inf')
    beta = float('inf')

    ordered_moves = _order_moves(state, moves, player)[:MAX_MOVES_TO_SEARCH]

    for move in ordered_moves:
        child = state.copy()
        _apply_move_for_search(child, move)

        score = _minimax(
            child, depth - 1, player,
            False,
            alpha, beta
        )

        move_scores[move] = score
        alpha = max(alpha, score)

    state.current_player = original_player
    return move_scores


def minimax_search_with_eval(
    state: GameState, depth: int, player: int, eval_fn
) -> dict[Move, float]:
    """same as minimax_search but uses a custom evaluation function."""

    def _minimax_custom(state, depth, root_player, is_maximizing,
                         alpha, beta):
        winner = check_winner(state)
        if winner == root_player:
            return 10000.0 + depth
        elif winner != 0 and winner is not None:
            return -10000.0 - depth

        if depth == 0:
            return eval_fn(state, root_player)

        moves = get_legal_moves(state, state.current_player)
        if not moves:
            return eval_fn(state, root_player)

        ordered_moves = _order_moves(state, moves, state.current_player)[:MAX_MOVES_TO_SEARCH]

        if is_maximizing:
            max_eval = float('-inf')
            for move in ordered_moves:
                child = state.copy()
                _apply_move_for_search(child, move)
                eval_score = _minimax_custom(
                    child, depth - 1, root_player, False, alpha, beta
                )
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in ordered_moves:
                child = state.copy()
                _apply_move_for_search(child, move)
                eval_score = _minimax_custom(
                    child, depth - 1, root_player, True, alpha, beta
                )
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval

    original_player = state.current_player
    state.current_player = player

    moves = get_legal_moves(state, player)
    move_scores = {}

    alpha = float('-inf')
    beta = float('inf')

    ordered_moves = _order_moves(state, moves, player)[:MAX_MOVES_TO_SEARCH]

    for move in ordered_moves:
        child = state.copy()
        _apply_move_for_search(child, move)

        score = _minimax_custom(
            child, depth - 1, player, False, alpha, beta
        )

        move_scores[move] = score
        alpha = max(alpha, score)

    state.current_player = original_player
    return move_scores
