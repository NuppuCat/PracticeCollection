"""
Tic Tac Toe Player
"""

import math
import copy
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
    if board==[[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]:
        s=X
        return s
    x = 0
    o = 0
    for i in range(3):
            
        for j in range(3):
            if board[i][j]==X:
                x=x+1
            elif  board[i][j]==O:
                o=o+1
    if x>o:
        s=O
    else:
        s=X
                
    
        
    return s


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    s = list()
    for i in range(3):
            
        for j in range(3):
            if board[i][j]==EMPTY:
                s.append([i,j])
    # print('actions',s)
    return s
# EMPTY = None
# board=[[EMPTY, EMPTY, EMPTY],
#             [EMPTY, EMPTY, EMPTY],
#             [EMPTY, EMPTY, EMPTY]]
# action = [0,1]
# i,j=action
# print(board[i][j]==EMPTY)

def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    ###################################################################################################
    #DEEP_COPY_iS_IMPORTANT!
    ###############################################################################################
    board_c = copy.deepcopy(board)
    i,j=action
    if board_c[i][j] != EMPTY:
        return board
    p=player(board_c)
    board_c[i][j] = p 
    
    return board_c
        

def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    for i in range(3):
        if board[i][0]==board[i][1] and board[i][1]== board[i][2] and board[i][0] != EMPTY:
            return board[i][0]
    for j in range(3):
        if board[0][j]==board[1][j] and board[1][j]== board[2][j] and board[0][j] != EMPTY:
            return board[0][j]    
    if board[0][0]==board[1][1] and  board[2][2]==board[1][1]  and  board[0][0] != EMPTY:
            return board[0][0]        
    if board[0][2]==board[1][1] and  board[2][0]==board[1][1]  and  board[0][2] != EMPTY:
            return board[0][2]         
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    b = False
    if winner(board)!=None:
        b = True
        return b
    for i in range(3):
            
        for j in range(3):
            if board[i][j]==EMPTY:
                return b
    b=True
    return b         
            
            


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    u = 0
    if winner(board)==X:
        u=1
    elif winner(board)==O:
        u=-1
    # raise NotImplementedError
    return u


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    
    """
    p = player(board)
    beastAction = None
    if  p==X:
                best_score = -1
                
                if terminal(board):
                    return beastAction
                set_of_actions = actions(board)
                for action in set_of_actions:
                    board_c = result(board, action)
                    # change player is important
                    score = getminimaxscore(O,board_c)
                    if score>best_score:
                        best_score=score
                        beastAction = action
    if  p==O:
                best_score = 1
                
                
                if terminal(board):
                    return beastAction
                set_of_actions = actions(board)
                for action in set_of_actions:
                    board_c = result(board, action)
                    score = getminimaxscore(X,board_c)
                    if score<best_score:
                        best_score=score
                        beastAction = action
    
        
    if  beastAction==None:
           beastAction=action        
    return beastAction  
            





def getminimaxscore(p, board_c):
    # score=-1
    if terminal(board_c):
        return utility(board_c)
        # if utility(board_c)==1 and p==X:
        #         score=1
        #         return score
        # elif utility(board_c)==-1 and p==O:
        #         score=1
        #         return score
        # elif utility(board_c)==0:
        #         score=0
        #         return score
        # else:
        #         score=-1
        #         return score    
    scores = []
    
    for action in actions(board_c):
        board_cc = result(board_c, action)
        # change player
        p_c = player(board_cc)
        scores.append(getminimaxscore(p_c, board_cc))
    # print('scores',scores)    
    if p==X:
        return max(scores)
    else:
        return min(scores)
        
     



# board_0 = [[EMPTY, EMPTY, EMPTY],
#           [EMPTY, X, EMPTY],
#           [EMPTY, EMPTY, EMPTY]] 
# print(terminal(board_0))        
# print(winner(board_0))   
# print(actions(board_0))  
# print(minimax(board_0))     
# print(board_0)      
        
        
        
        
        
        
        
        
        
        
        
        
        