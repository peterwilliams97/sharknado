#!/usr/bin/python
# -*- coding: utf-8 -*-
"""


"""
from __future__ import division
from collections import namedtuple
import os, glob, subprocess, re
import logging
import numpy as np
import sys

N = 8

def make_board():
    return np.zeros((N, N), dtype=np.int)


def propagate(queens, board):
    #print '----'
    #print list(enumerate(queens))
    board = board.copy()
    for x, y in enumerate(queens):
        board[x, :] = 1
        board[:, y] = 1
        
        xx, yy = x, y
        while xx >= 0 and yy >= 0:
            board[xx, yy] = 1
            xx -= 1
            yy -= 1    
        
        xx, yy = x, y
        while xx < N and yy >= 0:
            board[xx, yy] = 1
            xx += 1
            yy -= 1
            
        xx, yy = x, y    
        while xx >= 0 and yy < N:
            board[xx, yy] = 1
            xx -= 1
            yy += 1 
            
        xx, yy = x, y
        while xx < N and yy < N:
            board[xx, yy] = 1
            xx += 1
            yy += 1     
        
        if False:
            min_xy = min(x, y)
            max_xy = max(x, y)
            min_xy2 = min(x, N-y)
            max_xy2 = max(x, N-y)
            for i in range(-min_xy, N-max_xy):
                board[x + i, y + i] = 1
            for i in range(-min_xy2, max_xy2):
                try:
                    board[x + i, y - i] = 1 
                except:
                    pass
                    #print >> sys.stderr, min_xy2, max_xy2, i,  (x,x + i), (y,y - i)
    
    for x, y in enumerate(queens):
        board[x, y] = 8 
        
    
    #board[3,0] = 3
    #board[0,4] = 4
    #print board.T[::-1,:]         
    #print 
    return board            

final_queens = None
depth = 0
def add_queen(queens, board):
    global final_queens, depth
    
    if len(queens) > depth:
        depth = len(queens)
        print depth, N
        
    if len(queens) == N:
        print board.T[::-1,:] 
        print '!!!', list(enumerate(queens))
        final_queens = queens
        print '###', list(enumerate(final_queens))
        return True
        
    queens = queens[:]
    x = len(queens)
    for y in range(N):
        if board[x, y] != 0:
            #print '*', x, y
            continue
        #print '==============='
        #print x,y
        queens2 = queens + [y]
        board2 = propagate(queens2, board)
        if add_queen(queens2, board2):
            #print '@@@', queens2, x, y
            #print board2.T[::-1,:] 
            assert board2[x, y] == 8
            return True
            
    return False

    
def solve(n):
    global N, final_queens
    
    N = n
    queens = []    
    board = make_board()
    final_queens = None
    found = add_queen(queens, board)
    print found
    print final_queens
    return final_queens

if __name__ == '__main__':    
    N = int(sys.argv[1])
    sol = solve(N)
    print '-' * 80
    print sol
        
