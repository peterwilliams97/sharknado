#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import math, random
import numpy as np
from itertools import count
from utils import SortedDeque

DEBUG = False
EPSILON = 1e-6

def CLOSE(a, b):
    return abs(a - b) < EPSILON

random.seed(111)

def length(point1, point2):
    return np.hypot(point1, point2)
    #return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def _trip(locations, order):
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
   
    print 'populate_greedy', N, start
    assert 0 <= start < N
    all_nodes = set(range(N))
    order = np.empty(N, dtype=np.int32)
    order.fill(-1)
    #print order
    i = 0
    order[0] = start
    nodes = set([start])
    while i < N - 1:
        # print 'i=%d:' % i, all_nodes - nodes, len(nodes)
        if not (all_nodes - nodes):
            #print 'i=%d' % i
            break
        for j in closest[order[i]]:
            if j not in nodes:
                order[i + 1] = j
                nodes.add(j)
                i += 1
                break
    #print order            
    assert len(set(order)) == len(order), '%d %d, %d'  % (len(set(order)), len(order), j)

    return trip2(distances, order), order
    
     
def precalculate(points): 
    N = len(points)
    locations = np.array(points, dtype=np.float64)
    
    distances = np.zeros((N, N), dtype=np.float64)
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


def calc2opt(N, distances, order, dist_check):
    """2-opt
        Reverse [p1:p2] inclusive
    """

    assert isinstance(order, np.ndarray), type(order)
    assert CLOSE(dist_check, trip2(distances, order)), '%s %s' % (dist_check, trip2(distances, order))
    
    #print 'calc2opt:', order
    #print distances
    
    N1 = N - 1
   
    # select indices of two random points in the tour
    p1, p2 = random.randrange(0, N), random.randrange(0, N)
    # do this so as not to overshoot tour boundaries
   
    p1b = p1 - 1 if p1 > 0 else N1 
    p1a = p1 + 1 if p1 < N1 else 0
    w0, w1, w2 = order[p1b], order[p1], order[p1a]
        
    exclude = set([w0, w1, w2])
     
    while order[p2] in exclude:
        p2 = random.randrange(0, N)
    
    p2b = p2 - 1 if p2 > 0 else N1 
    p2a = p2 + 1 if p2 < N1 else 0
    w3, w4 = order[p2], order[p2a]
    
    # to ensure we always have p1<p2        
    if p2 < p1:
        p1, p2 = p2, p1
    
    p1a = p1 + 1 if p1 < N1 else 0
    p2a = p2 + 1 if p2 < N1 else 0
    w1, w2 = order[p1], order[p1a]   
    w3, w4 = order[p2], order[p2a]    
    #print (p1, p1a), (p2, p2a)    
    
    #print p1, p2, N1, order.shape
    #w1, w2 = order[p1], order[p1a]
   
    #print (p1, (w1, w2)), (p2, (w3, w4))
    #print (distances[w1, w2], distances[w3, w4]), (distances[w1, w3], distances[w2, w4])
    delta = (distances[w1, w3] + distances[w2, w4]) - (distances[w1, w2] + distances[w3, w4]) 
    
    if DEBUG:
        order1 = order.copy() 
        if p1a == 0:
            for i in range((p2+1)//2):
                order1[p2-i], order1[i] = order1[p2-i], order1[i] 
        else:
            order1[p1a:p2+1] = order[p2:p1a-1:-1] # reverse the tour segment between p1 and p2
        print order, trip2(distances, order)
        print order1, trip2(distances, order1)
        print delta
      
    return delta, p1a, p2 
    
    
    
def do2opt_best(N, distances, dist, order, max_iter):

    assert len(set(order)) == len(order)
    for _ in xrange(max_iter):
        delta, p1, p2 = calc2opt(N, distances, order, dist)
        if delta < 0:
            break
    
    assert dist + delta > 0    
    order1 = order.copy() # make a copy
    assert len(set(order1)) == len(order1), '%d %d : %d %d' % (len(set(order1)), len(order1), p1 , p2)
     
    #print 'order :', order.shape, order[p1:p2].shape 
    #print 'order1:', order1.shape,  order1[p2:p1:-1].shape 
 
    #print order1[p1:p2+1].shape, order[p2:p1-1:-1].shape
    #print order1[p1:p2+1]
    #print order1[p2:p1-1:-1]
    #print np.array(list(reversed(list(order1[p2:p1-1:-1]))))
    #print sorted(set(order1[p1:p2+1]))
    #print sorted(set(order[p2:p1-1:-1]))
    #print  order1[p1-1:p1+2], order1[p2-1:p2+2] 
    #print p1, p2, N, order1[p1:p2+1].shape, order[p2:p1-1:-1].shape
    if p1 == 0:
        for i in range((p2+1)//2):
            order1[p2-i], order1[i] = order1[p2-i], order1[i] 
    else:
        order1[p1:p2+1] = order[p2:p1-1:-1] # reverse the tour segment between p1 and p2  
    #print set(range(N)) - set(order1)
    assert len(set(order1)) == len(order1), '%d %d : %d %d' % (len(set(order1)), len(order1), p1 , p2)     
    
    if DEBUG:
        print p1, p2
        for o in order, order1:
            print '----'
            print '%s\n%s\n%s' % (o[:p1], o[p1:p2+1], o[p2+1:]) 
        
    assert CLOSE(dist, trip2(distances, order)), '%s %s' % (dist, trip2(distances, order))
 
    assert CLOSE(dist + delta, trip2(distances, order1)), '%s %s %s %s %e' % (dist, delta, 
        dist + delta,
        trip2(distances, order1),
        (dist + delta - trip2(distances, order1))/EPSILON)
    return dist + delta, order1 

    
def do2opt_any(N, distances, dist, order):
    
    assert CLOSE(dist, trip2(distances, order)), '%s %s' % (dist, trip2(distances, order))
    
    delta, p1, p2 = calc2opt(N, distances, order, dist)
    assert dist + delta > 0
    order1 = order.copy() # make a copy
    if p1 == 0:
        for i in range((p2+1)//2):
            order1[p2-i], order1[i] = order1[p2-i], order1[i] 
    else:
        order1[p1:p2+1] = order[p2:p1-1:-1] # reverse the tour segment between p1 and p2           
    assert isinstance(order, np.ndarray), type(order)
    #print '@12', order1.shape
    assert len(set(order1)) == len(order1), '%d %d' % (len(set(order1)), len(order1))     
    return dist + delta, order1


    
def calc3opt(N, distances, order, dist_check):
    """-opt
        Reverse [p1:p2] inclusive
    """

    assert isinstance(order, np.ndarray), type(order)
    assert CLOSE(dist_check, trip2(distances, order)), '%s %s' % (dist_check, trip2(distances, order))
    
    #print 'calc2opt:', order
    #print distances
    
    N1 = N - 1
   
    # select indices of 3 random points in the tour
    p1, p2, p3 = random.randrange(0, N), random.randrange(0, N), random.randrange(0, N)
    # do this so as not to overshoot tour boundaries
   
    p1b = p1 - 1 if p1 > 0 else N1 
    p1a = p1 + 1 if p1 < N1 else 0
    w0, w1, w2 = order[p1b], order[p1], order[p1a]
        
    exclude = set([w0, w1, w2])
     
    while order[p2] in exclude:
        p2 = random.randrange(0, N)
    
    p2b = p2 - 1 if p2 > 0 else N1 
    p2a = p2 + 1 if p2 < N1 else 0
    w3b, w3, w4 = order[p2b], order[p2], order[p2a]
    exclude.add(w3b)
    exclude.add(w3)
    exclude.add(w4)
    
    while order[p3] in exclude:
        p3 = random.randrange(0, N)
    
    # to ensure we always have p1 < p2 < p3       
    if p2 < p1: p1, p2 = p2, p1
    if p3 < p1: p1, p3 = p3, p1
    if p3 < p2: p2, p3 = p3, p2
    assert p1 < p2 < p3
    
    p1a = p1 + 1 if p1 < N1 else 0
    p2a = p2 + 1 if p2 < N1 else 0
    p3a = p3 + 1 if p3 < N1 else 0
    
    w1, w2 = order[p1], order[p1a]   
    w3, w4 = order[p2], order[p2a] 
    w5, w6 = order[p2], order[p2a]     
    #print (p1, p1a), (p2, p2a)    
    
    #print p1, p2, N1, order.shape
    #w1, w2 = order[p1], order[p1a]
   
    #print (p1, (w1, w2)), (p2, (w3, w4))
    #print (distances[w1, w2], distances[w3, w4]), (distances[w1, w3], distances[w2, w4])
    d0 = distances[w1, w2] + distances[w3, w4] + distances[w5, w6] # Original distance 
    d1 = distances[w1, w4] + distances[w2, w6] + distances[w3, w5] - d0
    d2 = distances[w1, w5] + distances[w2, w4] + distances[w3, w6] - d0
    d3 = distances[w1, w3] + distances[w2, w5] + distances[w4, w6] - d0
    d4 = distances[w1, w4] + distances[w2, w5] + distances[w3, w6] - d0 
        
    if DEBUG:
        order1 = order.copy() 
        if p1a == 0:
            for i in range((p2+1)//2):
                order1[p2-i], order1[i] = order1[p2-i], order1[i] 
        else:
            order1[p1a:p2+1] = order[p2:p1a-1:-1] # reverse the tour segment between p1 and p2
        print order, trip2(distances, order)
        print order1, trip2(distances, order1)
        print delta
      
    return (d1, d2, d3, d4), (p1, p2, p3)     

    
def reversed_order(order, p1, p2):
    order1 = np.empty[p2 - p1 + 1]
    if p1 == 0:
        for i in xrange(p2 + 1):
            order1[p2 - i] = order[i] 
    else:
        order1[p1:p2+1] = order[p2:p1-1:-1] # reverse the tour segment between p1 and p2
    

def do3opt_any(N, distances, dist, order):
    """
        a p1
        b p1a
        c p2
        d p2a
        e p3
        f p3a
    """
    
    assert CLOSE(dist, trip2(distances, order)), '%s %s' % (dist, trip2(distances, order))
    
    deltas, (p1, p2, p3), (p1a, p2a, p3a) = calc3opt(N, distances, order, dist)
    assert all((dist + d > 0) for d in deltas) 
    orders = [order.copy() for _ in deltas] # make copies
    
    # :a d:e c:b f:
    n = p1a 
    orders[0][n:n + p3-p2a+1] = order[p2a:p3+1]
    n += p3-p2a+1
    orders[0][n:n + p2-p1a+1] = reverse_order(order, p1a, p2)
    n += p3-p2a+1
    orders[0][n:N-p3a] = order[p3a:]
    assert CLOSE(dist+deltas[0], trip2(distances, orders[0]))
    assert len(set(orders[0])) == len(orders[0]), '%d %d' % (len(set(orders[0])), len(orders[0]))
    
    # :a e:d c:b f:
    n = p1a 
    orders[1][n:n + p3-p2a+1] = reverse_order(orders[0], p2a, p3)
    n += p3-p2a+1
    orders[1][n:n + p2-p1a+1] = order[p1a:p2+1]
    n += p2-p1a+1
    orders[1][n:N-p3a] = order[p3a:]
    assert CLOSE(dist+deltas[1], trip2(distances, orders[1]))
    
    # :a c:b e:d f:
    n = p1a 
    orders[2][n:n + p2-p1a+1] = order[p1a:p2+1]
    n += p2-p1a+1
    orders[2][n:n + p3-p2a+1] = reverse_order(orders[0], p2a, p3)
    n += p3-p2a+1
    orders[2][n:N-p3a] = order[p3a:]
    assert CLOSE(dist+deltas[2], trip2(distances, orders[2]))
     
    # :a d:e b:c f:
    n = p1a 
    orders[3][n:n + p3-p2a+1] = orders[p2a:p3+1]
    n += p3-p2a+1
    orders[3][n:n + p2-p1a+1] = order[p1a:p2+1]
    n += p2-p1a+1
    orders[3][n:N-p3a] = order[p3a:]
    assert CLOSE(dist+deltas[3], trip2(distances, orders[3]))
  
    return [dist + d for d in deltas], orders 

    
def do3opt_best(N, distances, dist, order):
    # Super inefficient !@#$
    dists, orders = do3opt_any(N, distances, dist, order)
    imin = min(list(enumerate(dists)), key=lambda x: x[1])
    return dists[imin], orders[imin]
    
    
def search(N, distances, visited, hash_base, dist, order):
    """Search for best solution starting with order"""
  
    assert isinstance(dist, float), type(dist)
    assert isinstance(order, np.ndarray), type(order)
    assert CLOSE(dist, trip2(distances, order))
    #print '@1', order.shape
  
    MAX_NO_IMPROVEMENT = 20 
    MAX_ITER = 40 
    MAX_NEIGHBORHOOD = 30 
    
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
                
            #+print ('@', dist),    
                
            hsh = np.dot(hash_base, order)
            #print hash_base.shape, order.shape, hsh.shape
            if hsh in visited:
                #print 'Seen this solution already'
                continue
            visited.add(hsh)        

            # Refine candidate solution using local search and neighborhood
            dist, order = do2opt_best(N, distances, dist, order, MAX_ITER)
            #if the cost of the candidate is less than cost of current best then replace
            #best with current candidate
            assert dist > 0
            if dist < best[0]:
                best, no_improvement_count = (dist, order), 0 # We also restart the search when we find the local optima
                # break: this breaks out of the neighborhoods iteration
                break
            #else: # increment the count as we did not find a local optima
            #    no_improvement_count +=1    
                       
            i = next(counter)   
            if i % 1000 == 100:
                print '$$', neighborhood, no_improvement_count, i
        else: # increment the count as we did not find a local optima
            no_improvement_count +=1        
                
        print '**', neighborhood, no_improvement_count, i, best[0]   
        visited.add(hsh)      

    print 'Done search', best    
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
    
    for start in xrange(N * 3):
        dist, order = populate_greedy(N, distances, closest, start//3)
        if start % 3 > 0:
            random.shuffle(order)
            dist = trip2(distances, order)
        assert len(set(order)) == len(order), start
        assert CLOSE(dist, trip2(distances, order)), '%s %s' % (dist, trip2(distances, order))
        
        dist, order = do2opt_best(N, distances, dist, order, MAX_ITER)
        assert CLOSE(dist, trip2(distances, order)), '%s %s' % (dist, trip2(distances, order))
        assert dist > 0
                       
        assert len(set(order)) == len(order), start
        hsh = np.dot(hash_base, order)
        #print hash_base.shape, order.shape, hsh.shape
        assert hsh not in visited
        visited.add(hsh)    
        outer_solutions.append((dist, hsh, order))
        if not optimum_solutions or dist < optimum_solutions[-1][0]:
            optimum_solutions.append((dist, hsh, order))
            print 'best:', optimum_solutions[-1][0]
            
    for dist, hsh, order in outer_solutions:
                       
        dist, order = search(N, distances, visited, hash_base, dist, order)
        assert dist > 0
        if dist < optimum_solutions[-1][0]:
            optimum_solutions.append((dist, hsh, order))
    
    dist, _, order = optimum_solutions.pop()
    
    print 'optimum:', dist, len(optimum_solutions), [x[0] for x in optimum_solutions[-10:]]
    
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
       
    # !@#$   
    points = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ] 

    points = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 0.5],
        [0.5, 0.0],
        [1.0, 0.5],
        [0.5, 1.0],
    ]  

    points = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        
        [0.0, 0.5],
        [0.5, 0.0],
        [1.0, 0.5],
        [0.5, 1.0],
        
        [0.0, 0.25],
        [0.25, 0.0],
        [1.0, 0.25],
        [0.25, 1.0],
    ]      
    nodeCount = len(points)

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
    print dist
    print [points[i] for i in order]
        
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

