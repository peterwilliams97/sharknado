#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    Solve the N queens problem http://en.wikipedia.org/wiki/Eight_queens_puzzle
    using naive constraint propagation.
    
    Usage: 
        python n_queens.py <N>
    e.g.
        python .n_queens.py 20
        
    NOTES:
        Uses NumPy and Numba so the best way to run this is to get the latest Anaconda disto
        from https://store.continuum.io/
        If you don't want to use Numba then remove the @autojit. 
        If you don't want to even use Numpy then make board a list of lists.
"""
from __future__ import division
import numpy as np
import sys, random
from numba import autojit


def make_board(N):
    """Create a N x N chessboard
        Returns: An N X N integer NumPy array
    """    
    return np.zeros((N, N), dtype=np.int)
    

@autojit
def propagate(N, board, x, y, d):
    """ Propagate the constraints of adding (d = +1) or removing (d = -1) a queen to the board at 
        (x, y)
        N: Width of chessboard
        board: State of chessboard for current queens which does not include the one at (x, y). 
               board[x, y] = 1 if (x, y) is threatened by any of queens
        x, y: Coordinates of queen to be added/removed
           d: +1 to add a queen, -1 to remove a queen
    """

    # Add d to the elements in the column and row that intersect (x, y) including (x, y)
    board[x, :] += d
    board[:, y] += d
    board[x, y] -= d

    # Add d to the elements in the diagonals that intersect (x, y) excluding (x, y)
    # This looks at a bit funny because of the current limitations of Numba but this is the 
    #  inner loop of this script so I tolerate the funniness.
    x1 = x - 1
    y1 = y - 1
    x2 = x + 1
    y2 = y - 1
    x3 = x - 1
    y3 = y + 1
    x4 = x + 1
    y4 = y + 1       

    while x1 >= 0 and y1 >= 0:
        board[x1, y1] += d
        x1 -= 1
        y1 -= 1    

    while x2 < N and y2 >= 0:
        board[x2, y2] += d
        x2 += 1
        y2 -= 1
      
    while x3 >= 0 and y3 < N:
        board[x3, y3] += d
        x3 -= 1
        y3 += 1 
   
    while x4 < N and y4 < N:
        board[x4, y4] += d
        x4 += 1
        y4 += 1     


def add_queen(N, queens, board):
    """Add a queen to board
        Board is filled from left column to right.
        This function called recursively. 
        It gets called with a valid list of queens and attempts to add one more queen. 
        
        N: Width of chessboard
        queens: List of queens that have been placed so far. queens[i] = row of queen on column i
        board: State of chessboard. board[x, y] = 1 if (x, y) is threatened by any of queens
        Returns: Valid list of N queens if one has been found, None otherwise
        
        Note: board gets modified then restored to its state at entry to this function.
    """
    # The queens are valid by design. If there are N of them so we must have a full solution.
    if len(queens) == N:
        return queens

    # Checking rows in random order makes this run fast
    row_order = list(range(N))
    random.shuffle(row_order) 
    
    x = len(queens)
    for y in row_order:
        # Don't check threatened positions. This is the constraint that makes this work.
        if board[x, y] != 0:
            continue
        
        # Propagate the contraints that are added by the queen just added at (x, y)
        propagate(N, board, x, y, 1)
        # Recurse
        valid_queens = add_queen(N, queens + [y], board)
        # Undo the constraint added for queen at x, y
        propagate(N, board, x, y, -1)
        
        # If we found a valid board deeper in the recursion just return it.
        if valid_queens:
            return valid_queens
            
    # If we got here then queens could be not be extended to a valid N queens lits 
    return None
    
    
def solve(N):
    """Solve the N queens problem
        queens = solve(N) => queens[i] is the row of the queen in column i
                           where 0 <= row < N and 0 <= column < N
        Returns: List of rows for queens. 
    """
    return add_queen(N, [], make_board(N))

    
if __name__ == '__main__':    
    if len(sys.argv) != 2:
        print __doc__
        exit()
    N = int(sys.argv[1])
    print 'Solving %d queens problem' % N
    queens = solve(N)
    print queens
    if N <= 80:
        print '-' * 80
        for y in range(N):
            print ''.join(' Q' if x == y else ' +' for x in queens) 
        
