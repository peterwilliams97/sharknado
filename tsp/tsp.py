#!/usr/bin/python
# -*- coding: utf-8 -*-
import math
import numpy as np
from utils import SortedDeque

def length(point1, point2):
    return np.hypot(point1, point2)
    #return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def trip(locations, order):
    dist = 0
    p1 = locations[order[0]]
    for i in order[1:]:
        p2 = locations[i]
        #dist += np.hypot(p1 - p2) 
        diff = p1 - p2
        dist += np.sqrt(diff[0] ** 2 +diff[1] ** 2) 
        #print '*', p1, p2, dist
        p1 = p2
        #exit()    
    return dist
  
NUM_SOLUTIONS = 40  
        
def solve(points):
    """Return traversal order of points that minimizes distance travelled"""
    
    N = len(points)
    locations = np.array(points, dtype=np.float32)
    order = np.arange(N)
    
    distances = np.zeros((N, N), dtype=np.float32)
    for i in xrange(N):
        diff = locations - locations[i]
        #print diff.shape
        for j in xrange(N):
            #print diff[j].shape
            #print np.sqrt(diff[0] ** 2 + diff[1] ** 2).shape
            distances[i][j] = np.sqrt(diff[j][0] ** 2 + diff[j][1] ** 2) #;  np.hypot(diff[j]) 
 
        
    hash_base = np.random.randint(10**4, 10**6, N)
    visited = set()
    #hsh = np.dot(hash_base, order)
    #dist = trip(locations, order) 
    #best = dist
    #    if hsh in visited: 
    #        continue
        
    
    solutions = SortedDeque([], maxlen=NUM_SOLUTIONS)
    for count in xrange(NUM_SOLUTIONS * 100):
        hsh = np.dot(hash_base, order)
        if hsh in visited: 
            continue
        visited.add(hsh)    
        dist = trip(locations, order)
        #print dist
        solutions.insert((dist, hsh, order))
        np.random.shuffle(order)
        if count >= NUM_SOLUTIONS * 1 and len(solutions) == NUM_SOLUTIONS:
            break
    
    
        
    eps = 1e-1
    eps2 = eps * 1.1
    e = np.exp(1.0)
    counts = np.zeros(NUM_SOLUTIONS)
    for __ in xrange(4):
        for _ in xrange(NUM_SOLUTIONS * 1000):
            #x = np.random.random()
            #y = -np.log((x + eps)/(1.0 + eps))
            #z = y * NUM_SOLUTIONS-1
            z = np.random.exponential(NUM_SOLUTIONS * 0.1)
            n = int(np.floor(z))
            #print (x, y, z), (n, NUM_SOLUTIONS)
            if n >= NUM_SOLUTIONS: n = NUM_SOLUTIONS -1
            counts[n] +=1     
            #continue
            dist, hsh, order = solutions[n]
            i1 = np.random.randint(0, N - 1)
            i2 = np.random.randint(0, N - 1)
            if i1 == i2:
                continue
            order2 = order.copy()
            order2[i1] = order[i2]
            order2[i2] = order[i1]
            dist = trip(locations, order2)
            hsh = np.dot(hash_base, order2)
            if hsh in visited: 
                continue
            visited.add(hsh)    
            solutions.insert((dist, hsh, order2))
        
        #print counts
        dist, hsh, order = solutions[0]
        print dist # , hsh, order
    
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

