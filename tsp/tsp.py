#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import math, random
import numpy as np
from itertools import count
from utils import SortedDeque

def length(point1, point2):
    return np.hypot(point1, point2)
    #return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def trip(locations, order):
    dist = 0
    p1 = locations[order[-1]]
    for i in order:
        p2 = locations[i]
        #dist += np.hypot(p1 - p2) 
        diff = p1 - p2
        dist += np.sqrt(diff[0] ** 2 +diff[1] ** 2) 
        #print '*', p1, p2, dist
        p1 = p2
        #exit()    
    return dist


def trip2(distances, order):
    dist = 0
    i = order[-1]
    for j in order:
        dist += distances[i, j]  
        i = j    
    return dist
    
    
def populate_greedy(N, distances, closest, start):
    traverse = np.zeros(N)
    traverse[start:] = np.arange(N - start)
    traverse[:start] = N - start + np.arange(start)
    
    order = np.arange(N)
    order[0] = start
    nodes = set()
    for i in traverse:
        for j in closest[i]:
            if j not in nodes:
                order[i] = j
                nodes.add(j)

    return trip2(distances, order), order
    
     
def precalculate(points): 
    N = len(points)
    locations = np.array(points, dtype=np.float32)
    
    
    distances = np.zeros((N, N), dtype=np.float32)
    for i in xrange(N):
        diff = locations - locations[i]
        #print diff.shape
        for j in xrange(N):
            #print diff[j].shape
            #print np.sqrt(diff[0] ** 2 + diff[1] ** 2).shape
            distances[i][j] = np.sqrt(diff[j][0] ** 2 + diff[j][1] ** 2) #;  np.hypot(diff[j]) 
            
    # Ordering of distances, closest firts.
    closest = np.zeros((N, N), dtype=np.int32)
    for i in xrange(N):
        a = range(N)
        a.sort(key=lambda j: distances[i, j])
        closest[i, :] = a 

    return N, distances, closest  


def calc2opt(N, distances, order):

    assert isinstance(order, np.ndarray), type(order)
        
    N1 = N - 1
   
    # select indices of two random points in the tour
    p1, p2 = random.randrange(0, N), random.randrange(0, N)
    # do this so as not to overshoot tour boundaries
    p1b = p1 - 1 if p1 > 0 else N1 
    p1a = p1 + 1 if p1 < N1 else 0
    exclude = set([p1, p1b, p1a])
     
    while p2 in exclude:
        p2 = random.randrange(0, N)
    
    p2b = p2 - 1 if p2 > 0 else N1 
    p2a = p2 + 1 if p2 < N1 else 0

    # to ensure we always have p1<p2        
    if p2 < p1:
        p1, p2 = p2, p1
    
    #print p1, p2, N1, order.shape
    w1, w2 = order[p1a], order[p1]
    w3, w4 = order[p2a], order[p2]
    delta = distances[w1, w2] + distances[w3, w4] - (distances[w1, w3] + distances[w2, w4])
    return delta, p1, p2 
    
    
def do2opt_best(N, distances, dist, order, max_iter):
    for _ in xrange(max_iter):
        delta, p1, p2 = calc2opt(N, distances, order)
        if delta < 0:
            break
            
    order1 = order.copy() # make a copy
    #print 'order :', order.shape, order[p1:p2].shape 
    #print 'order1:', order1.shape,  order1[p2:p1:-1].shape 
    order1[p1:p2] = order[p2:p1:-1] # reverse the tour segment between p1 and p2  
    assert len(set(order1)) == len(order1), '%d %d : %d %d' % (len(set(order1)), len(order1), p1 , p2)     
    return dist + delta, order1 

    
def do2opt_any(N, distances, dist, order):
    delta, p1, p2 = calc2opt(N, distances, order)
    order1 = order.copy() # make a copy
    order1[p1:p2] = order[p2:p1:-1] # reverse the tour segment between p1 and p2       
    assert isinstance(order, np.ndarray), type(order)
    #print '@12', order1.shape
    assert len(set(order1)) == len(order1), '%d %d' % (len(set(order1)), len(order1))     
    return dist + delta, order1
    
    
def search(N, distances, visited, hash_base, dist, order):
    """Search for best solution starting with order"""
  
    assert isinstance(dist, float), type(dist)
    assert isinstance(order, np.ndarray), type(order)
    #print '@1', order.shape
  
    MAX_NO_IMPROVEMENT = 10 
    MAX_ITER = 20 
    MAX_NEIGHBORHOOD = 20 
    
    local_min_count = 0
    
    best = (dist, order)
    
    print 'search', dist
    
    no_improvement_count = 0
    counter = count()
    i = next(counter)   
    while no_improvement_count <= MAX_NO_IMPROVEMENT:
        
        # for each neighborhood in neighborhoods
        for neighborhood in range(1, MAX_NEIGHBORHOOD):
            
            #Calculate Neighborhood : Involves running stochastic two opt for neighbor times
            for index in range(0, neighborhood):
                # Get candidate solution from neighborhood
                dist, order = do2opt_any(N, distances, dist, order)
                #print '@2', order.shape
                
            #print '@3', order.shape    
                
            hsh = np.dot(hash_base, order)
            #print hash_base.shape, order.shape, hsh.shape
            if hsh in visited:
                #print 'Seen this solution already'
                continue
            visited.add(hsh)        

            # Refine candidate solution using local search and neighborhood
            dist, order = do2opt_best(N, distances, dist, order, MAX_ITER)
            #if the cost of the candidate is lesser than cost of current best then replace
            #best with current candidate
            if dist < best[0]:
                best, no_improvement_count = (dist, order), 0 # We also restart the search when we find the local optima
                # break: this breaks out of the neighborhoods iteration
                break
            else: # increment the count as we did not find a local optima
                no_improvement_count +=1
               
            i = next(counter)   
            if i % 1000 == 100:
                print '$$', neighborhood, no_improvement_count, i
                
        #print '**', neighborhood, no_improvement_count, i   
        visited.add(hsh)      

    return best                
    
NUM_SOLUTIONS = 40 
MAX_ITER = 100 
        
def solve(points):
    """Return traversal order of points that minimizes distance travelled"""
    
    N, distances, closest = precalculate(points)
        
    hash_base = np.random.randint(10**4, 10**6, N)
    visited = set()
    
    outer_solutions = []
    optimum_solutions = []
    
    for start in xrange(N):
        dist, order = populate_greedy(N, distances, closest, start)
        delta, order = do2opt_best(N, distances, dist, order, MAX_ITER)
        hsh = np.dot(hash_base, order)
        #print hash_base.shape, order.shape, hsh.shape
        assert hsh not in visited
        visited.add(hsh)    
        outer_solutions.append((dist, hsh, order))
        if delta < 0 or not optimum_solutions:
            optimum_solutions.append((dist, hsh, order))
            print 'best:', optimum_solutions[-1][0]
            
    for dist, hsh, order in outer_solutions:
                       
        delta, order = search(N, distances, visited, hash_base, dist, order)
        if delta < 0:
            optimum_solutions.append((dist, hsh, order))
    
    dist, _, order = optimum_solutions.pop()
    return dist, list(order)
    
def solveIt(inputData):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = inputData.split('\n')

    nodeCount = int(lines[0])

 
    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append((float(parts[0]), float(parts[1])))

    if False:    
        # build a trivial solution
        # visit the nodes in the order they appear in the file
        solution = range(0, nodeCount)

        # calculate the length of the tour
        obj = length(points[solution[-1]], points[solution[0]])
        for index in range(0, nodeCount-1):
            obj += length(points[solution[index]], points[solution[index+1]])

    assert nodeCount == len(points)        
    dist, order = solve(points)    
        
    # prepare the solution in the specified output format
    outputData = str(dist) + ' ' + str(0) + '\n'
    outputData += ' '.join(map(str, order))

    return outputData


import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        fileLocation = sys.argv[1].strip()
        inputDataFile = open(fileLocation, 'r')
        inputData = ''.join(inputDataFile.readlines())
        inputDataFile.close()
        print solveIt(inputData)
    else:
        print 'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)'

