"""
Tic Tac Toe Player
"""

import copy
import random

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    
    flat_board = [val for row in board for val in row]
    
    return O if flat_board.count(X) > flat_board.count(O) else X
    

def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """

    actions = set()
    
    for i, row in enumerate(board):
        for j, val in enumerate(row):
            if val == EMPTY:
                actions.add((i,j))
                
    return actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    
    if action not in actions(board):
        raise ValueError
    
    board_c = copy.deepcopy(board)
    board_c[action[0]][action[1]] = player(board_c)
    
    return board_c


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """

    conditions = [[(0,0), (1,0), (2,0)], [(0,1), (1,1), (2,1)], [(0,2), (1,2), (2,2)],
                  [(0,0), (0,1), (0,2)], [(1,0), (1,1), (1,2)], [(2,0), (2,1), (2,2)], 
                  [(0,0), (1,1), (2,2)], [(2,0), (1,1), (0,2)]]
    
    
    for sol in conditions:
        if len(set([mark(board, action) for action in sol])) == 1 and mark(board, sol[0]) != EMPTY:
            return mark(board, sol[0])
    
    return None
        
def mark(board, action):
    """
    Returns X, O or None based on the board and the action
    """
    
    return board[action[0]][action[1]]
    

def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    
    return True if winner(board) is not None or not actions(board) else False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    
    return 1 if winner(board) == X else (-1 if winner(board) == O else 0)


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None

    # Making the first move -> optimal solution is to go to corner since then only
    # by taking center the loss can be prevented by the second player. However, by
    # choosing corner it is not possible to win if opponent also playes optimally.
    # This is the case for all starting choises -> winning is not possible
    if board == initial_state():
        corners = [(0,0), (2,0), (0,2), (2,2)]
        return random.choice(corners)

    current_player = player(board)
    best_value = float("-inf") if current_player == X else float("inf")

    for action in actions(board):
        new_value = minimax_value(result(board, action), best_value)
        new_value = max(best_value, new_value) if current_player == X else min(best_value, new_value)

	# This happens atleast once -> best_action is defined
        if new_value != best_value:
            best_value = new_value
            best_action = action

    return best_action


def minimax_value(board, best_value):
    """
    Returns the best value using recursive minimax evaluation. 
    Solution uses alpha-beta pruning which is why the new best value is returned
    without checking the other "branches".
    """
    if terminal(board):
        return utility(board)

    value = float("-inf") if player(board) == X else float("inf")

    for action in actions(board):
        new_value = minimax_value(result(board, action), value)
        
        value = max(value, new_value) if player(board) == X else min(value, new_value)
        if player(board) == X and new_value > best_value: return new_value 
        elif player(board) != X and new_value < best_value: return new_value

    return value
