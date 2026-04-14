"""
Phase 2: Minimax Search

Minimax with alpha-beta pruning.
Searches a fully observable game tree.
"""

import sys
# Make sure we don't hit recursion limits 
sys.setrecursionlimit(2000)

from shared.types import GameState, Move
from game.moves import get_legal_moves
from game.game_loop import apply_move
from game.win_detection import check_winner
from agent.heuristic import evaluate, POSITION_VALUES

MAX_MOVES_TO_SEARCH = 30

def _order_moves(state: GameState, moves: list[Move], player: int) -> list[Move]:
    """Sort moves so promising ones are searched first."""
    def move_priority(move: Move) -> int:
        r, c = move.position
        if move.move_type == "place" or move.move_type == "wild":
            # Prefer moves near existing friendly chips and on high-value positions
            return int(POSITION_VALUES[r, c])
        elif move.move_type == "remove":
            # Prefer removing chips that are part of threats
            return 50  # removing is usually high priority
        return 0
    
    return sorted(moves, key=move_priority, reverse=True)

def _minimax(
    state: GameState, 
    depth: int, 
    root_player: int, 
    is_maximizing: bool, 
    alpha: float, 
    beta: float
) -> float:
    """Recursive helper for minimax_search."""
    # Base case: leaf node or game over
    winner = check_winner(state)
    if winner == root_player:
        return 10000.0 + depth  # win sooner = higher score
    elif winner != 0 and winner is not None:
        return -10000.0 - depth # lose = very negative
    
    if depth == 0:
        return evaluate(state, root_player)
    
    # Needs to know whose turn it is
    moves = get_legal_moves(state, state.current_player)
    if not moves:
        return evaluate(state, root_player)
        
    ordered_moves = _order_moves(state, moves, state.current_player)[:MAX_MOVES_TO_SEARCH]
    
    if is_maximizing:
        max_eval = float('-inf')
        for move in ordered_moves:
            child = state.copy()
            # apply_move draws a card, adding some noise
            apply_move(child, move) 
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
            apply_move(child, move)
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
    Run minimax with alpha-beta pruning on a fully observable state.
    
    Parameters:
      state  - a complete GameState
      depth  - plies to search
      player - the player we are maximizing for
      
    Returns:
      dict mapping each legal Move to its minimax score.
    """
    # Note: State current_player is expected to be `player` initially, but minimax_search
    # is called from the perspective of root `player`.
    
    # Save the original current_player just in case, though it should be `player`
    original_player = state.current_player
    state.current_player = player
    
    moves = get_legal_moves(state, player)
    move_scores: dict[Move, float] = {}
    
    alpha = float('-inf')
    beta = float('inf')
    
    ordered_moves = _order_moves(state, moves, player)[:MAX_MOVES_TO_SEARCH]
    
    for move in ordered_moves:
        child = state.copy()
        apply_move(child, move)
        
        score = _minimax(
            child, depth - 1, player,
            False,  # Next level is minimizing
            alpha, beta
        )
        
        move_scores[move] = score
        alpha = max(alpha, score)
        
    # Restore just in case
    state.current_player = original_player
        
    return move_scores
