#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Sovlve the n queens problem http://en.wikipedia.org/wiki/Eight_queens_puzzle

"""
from __future__ import division
from collections import namedtuple
import os, glob, subprocess, re
import logging
import numpy as np
import sys, random
from numba import autojit

N = 8

def make_board():
    return np.zeros((N, N), dtype=np.int)

@autojit
def propagate(board, x, y, d):
    
    board[x, :] += d
    board[:, y] += d
    board[x, y] -= d

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
          

order = list(range(N))
#random.shuffle(order) 
   
final_queens = None
depth = 0
counts = {}

def add_queen(queens, board):
    global final_queens, depth
    
    if len(queens) >= depth:
        print len(queens), depth, N
        depth = len(queens)
        
        
    if len(queens) == N:
        print board.T[::-1,:] 
        print '!!!', list(enumerate(queens))
        final_queens = queens
        print '###', list(enumerate(final_queens))
        return True
        
    queens = queens[:]
    x = len(queens)
    order2 = order[:]
    random.shuffle(order2) 
    for y in order2:
        if board[x, y] != 0:
            continue
        queens2 = queens + [y]
        propagate(board, x, y, 1)
        found = add_queen(queens2, board)
        propagate(board, x, y, -1)
        if found:
            #print '@@@', queens2, x, y
            #print board2.T[::-1,:] 
            #assert board[x, y] == 8
            return True
            
    return False

    
    
def solve(n):
    global N, order, final_queens
    
    soln = '%d.soln' % n
    if os.path.exists(soln):
        print 'reading existing'
        with open(soln, 'rt') as f:
            line = f.readline().strip('\n')
            final_queens = eval(line)
    else:        
        N = n
        order = list(range(N))
        random.shuffle(order) 
        queens = []    
        board = make_board()
        final_queens = None
        found = add_queen(queens, board)
        print found
        print final_queens
        with open(soln, 'wt') as f:
            f.write('%s\n' % repr(final_queens))
    return final_queens

if __name__ == '__main__':    
    N = int(sys.argv[1])
    sol = solve(N)
    print '-' * 80
    print sol
        
