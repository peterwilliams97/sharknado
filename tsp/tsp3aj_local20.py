#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import math, random
import numpy as np
from itertools import count
from utils import SortedDeque

from numba import autojit, jit, double

VERSION = 3
print 'VERSION=%d' % VERSION

random.seed(111)

DEBUG = False
EPSILON = 1e-6

def CLOSE(a, b):
    return abs(a - b) < EPSILON
 
def DIFF(a, b): 
    return '%s - %s = %s' % (a, b, a -b)    

random.seed(111)

@autojit
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
   
    #print 'populate_greedy', N, start
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
    

hack_locations = None    
def precalculate(points): 
    global hack_locations
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
            
    # Ordering of distances, closest first.
    closest = np.zeros((N, N-1), dtype=np.int32)
    for i in xrange(N):
        a = range(N)
        a.sort(key=lambda j: distances[i, j])
        closest[i, :] = a[1:] 

    hack_locations = locations    
    return N, locations, distances, closest 
    
    
def normalize(N, order):
    """For any closed tour there are 2N orders
        N rotations x (forward|reverse)
        We choose one of them here
    """    
    for i in xrange(N):
        if order[i] == 0:
            break
    order1 = np.roll(order, -i) 
    if order1[1] < order1[-1]:
        order[:] = order1[:]
    else:
        order[0] = 0
        order[1:] = order1[1:][::-1]
    assert order[0] == 0, '%d: %s' % (i, order)    


import matplotlib.pyplot as plt


def draw_path(order, boundaries=None):

    if boundaries:
        (p1, p2, p3), (p1a, p2a, p3a) = boundaries
        b_dict = { p1:'p1', p1a:'p1a', p2:'p2', p2a:'p2a', p3:'p3', p3a:'p3a'}
    else: 
        b_dict = {}
    
    locations = hack_locations
    locations2 = np.zeros(locations.shape)
    for i in order:
        locations2[i,:] = locations[order[i],:]
    
    print locations
    print order
    print locations2
    
    x1 = []
    y1 = []
    delta = 0.0
    for i in order:
        x1.append(locations[i,0] - delta)
        y1.append(locations[i,1] - delta)
        delta -= 0.005
    x1.append(locations[order[0], 0])
    y1.append(locations[order[0], 1])
    plt.xlim((-.1, 1.2)) 
    plt.ylim((-.1, 1.2)) 
    plt.plot(x1, y1, marker='x', linestyle = '-', color = 'b')
    for i, o in enumerate(order):
        plt.text(x1[i], y1[i] + 0.01, '%d %s' % (i, b_dict.get(i, '')))
    
 
def show_path(order):
    draw_path(order)
    plt.show() 
    
def show_path2(dist, order, dist1, order1, boundaries):
    return
    plt.subplot(211)
    plt.title('before: %.2f' % dist)
    draw_path(order, boundaries)
    plt.subplot(212)
    plt.title('after: %.2f' % dist1)
    draw_path(order1, boundaries)
    plt.show() 

HISTORY = 'history%02d.py'   
saved_points = {}    
saved_solutions = {}    
def save_solution(path, points, dist, order):
    saved_points[path] = points
    saved_solutions[path] = (dist, order) 
    with open(HISTORY % VERSION, 'wt') as f:
        f.write('VERSION=%d\n' % VERSION)
        f.write('saved_solutions = %s\n' % repr(saved_solutions))
        f.write('saved_points = %s\n' % repr(saved_points))
    
@autojit
def calc2opt_delta(N, distances, order, dist_check, boundary_starts):
    N1 = N - 1
    p1, p2 = boundary_starts

    p1a = p1 + 1 if p1 < N1 else 0
    p2a = p2 + 1 if p2 < N1 else 0
    w1, w2 = order[p1], order[p1a]   # a b
    w3, w4 = order[p2], order[p2a]   # c d 
    delta = (distances[w1, w3] + distances[w2, w4]) - (distances[w1, w2] + distances[w3, w4])
    return delta, ((p1, p2), (p1a, p2a)) 
    
def calc2opt(N, distances, order, dist_check):
    """2-opt
        Reverse [p1:p2] inclusive
    """

    #assert isinstance(order, np.ndarray), type(order)
    #assert CLOSE(dist_check, trip2(distances, order)), '%s %s' % (dist_check, trip2(distances, order))
    
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
      
    return delta, (p1a, p2) 


def _do2opt_best(N, distances, dist, order, max_iter):

    assert len(set(order)) == len(order)
    for _ in xrange(max_iter):
        delta, (p1, p2) = calc2opt(N, distances, order, dist)
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

@autojit
def reverse_order(order, p1, p2):
    """Return order[p1:p2+1] reversed"""
    order1 = np.empty(p2 - p1 + 1)
    #print 'reverse_order:', order.shape, p1, p2, order1.shape
    if p1 == 0:
        for i in xrange(p2 + 1):
            order1[p2 - i] = order[i] 
    else:
        order1[:] = order[p2:p1-1:-1] # reverse the tour segment between p1 and p2
    return order1    
   
@autojit   
def do2opt(N, distances, dist, order, delta, p1, p2, p1a, p2a): 
    #(p1, p2), (p1a, p2a) = boundaries
    order1 = order.copy() # make a copy
    
    #print (p1, p2), N, order
    #t = reverse_order(order, p1, p2)
    #print order1[p1:p2+1].shape, t.shape
    order1[p1a:p2+1] = reverse_order(order, p1a, p2)
    #dist1 = dist + delta
    #show_path2(dist, order, dist1, order1, boundaries)
    #assert CLOSE(dist1, trip2(distances, order1)), DIFF(dist1, trip2(distances, order1)) 
    #assert len(set(order1)) == len(order1), '%d %d' % (len(set(order1)), len(order1))     
    return dist + delta, order1  
    
def _do2opt_any(N, distances, dist, order):
    
    #assert CLOSE(dist, trip2(distances, order)), '%s %s' % (dist, trip2(distances, order))
    
    p1, p2 = random.randrange(0, N), random.randrange(0, N)
    # do this so as not to overshoot tour boundaries
 
    N1 = N - 1
    p1b = p1 - 1 if p1 > 0 else N1 
    p1a = p1 + 1 if p1 < N1 else 0
    w0, w1, w2 = order[p1b], order[p1], order[p1a]
        
    exclude = set([w0, w1, w2])
     
    while order[p2] in exclude:
        p2 = random.randrange(0, N)
        
    # to ensure we always have p1<p2        
    if p2 < p1:
        p1, p2 = p2, p1


    delta, boundaries = calc2opt_delta(N, distances, order, dist, (p1, p2))
    #assert dist + delta > 0
    
    #print '%', boundaries
    return do2opt(N, distances, dist, order, delta, p1, p2, p1a, p2a)
    
    order1 = order.copy() # make a copy
    if p1 == 0:
        for i in range((p2+1)//2):
            order1[p2-i], order1[i] = order1[p2-i], order1[i] 
    else:
        order1[p1:p2+1] = order[p2:p1-1:-1] # reverse the tour segment between p1 and p2           
    #assert isinstance(order, np.ndarray), type(order)
    #print '@12', order1.shape
    #assert len(set(order1)) == len(order1), '%d %d' % (len(set(order1)), len(order1))     
    return dist + delta, order1

@autojit
def calc3opt_deltas(N, distances, order, dist_check, boundary_starts):    
    N1 = N - 1
    (p1, p2, p3) = boundary_starts

    p1a = p1 + 1 if p1 < N1 else 0
    p2a = p2 + 1 if p2 < N1 else 0
    p3a = p3 + 1 if p3 < N1 else 0
    
    w1, w2 = order[p1], order[p1a]   # a b
    w3, w4 = order[p2], order[p2a]   # c d  
    w5, w6 = order[p3], order[p3a]   # e f  

    #ww = (w1, w2, w3, w4, w5, w6)
    #assert len(set(ww)) == len(ww)
    
    bf = distances[w1, w2] + distances[w3, w4] + distances[w5, w6] # Original distance 
    d0 = distances[w1, w4] + distances[w2, w6] + distances[w3, w5] - bf
    d1 = distances[w1, w5] + distances[w2, w4] + distances[w3, w6] - bf
    d2 = distances[w1, w3] + distances[w2, w5] + distances[w4, w6] - bf
    d3 = distances[w1, w4] + distances[w2, w5] + distances[w3, w6] - bf 
   
    deltas = (d0, d1, d2, d3)
    
    #if DEBUG:
    #   print 'deltas:', deltas
    #   print 'boundaries:', ((p1, p2, p3), (p1a, p2a, p3a))
    #   print 'w:', (w1, w2), (w3, w4), (w5, w6) 
    #   print 'bf:', bf, (distances[w1, w2], distances[w3, w4], distances[w5, w6])
    #   print 'd2:', d2 + bf, (distances[w1, w3], distances[w2, w5], distances[w4, w6]), d2
    #   #show_path(order)

    return deltas, ((p1, p2, p3), (p1a, p2a, p3a))  
    
def calc3opt(N, distances, order, dist_check):
    """3-opt
        
    """

    #assert isinstance(order, np.ndarray), type(order)
    #assert CLOSE(dist_check, trip2(distances, order)), '%s %s' % (dist_check, trip2(distances, order))
    
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
    #exclude.add(order[N1])
     
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
    #assert p1 < p1a < p2 < p2a < p3 < p3a, ((p1, p1a), (p2, p2a), (p3, p3a)) 
        
    boundary_starts = (p1, p2, p3)
    deltas, boundaries = calc3opt_deltas(N, distances, order, dist_check, boundary_starts)
    #dist1, order1 = do3opt_2(N, distances, dist_check, order, deltas, boundaries)
    
    return deltas, boundaries     

    
1

#        a p1
#        b p1a
#        c p2
#        d p2a
#        e p3
#        f p3a    


def do3opt_0(N, distances, dist, order, deltas, boundaries):        
    (p1, p2, p3), (p1a, p2a, p3a) = boundaries
    order1 = order.copy()
    
    #print 'do3opt_0:', N, (p1, p2, p3), (p1a, p2a, p3a)
    
    # :a d:e c:b f:
    #print ':%d %d:%d %d:%d %d:' % (p1, p2a, p3, p2, p1a, p3a)
    n = p1a 
    order1[n:n + p3-p2a+1] = order[p2a:p3+1]
    n += p3-p2a+1
    order1[n:n + p2-p1a+1] = reverse_order(order, p1a, p2)
    n += p2-p1a+1
    if n < N and p3a != 0:
        #print n, N, p3a
        order1[n:N] = order[p3a:]
    
    if DEBUG:
        print np.arange(N), 'indexes'
        print order, trip2(distances, order)
        print order1, trip2(distances, order1)
    dist1 = dist + deltas[0]
    show_path2(dist, order, dist1, order1, boundaries)
    #assert CLOSE(dist1, trip2(distances, order1)), DIFF(dist1, trip2(distances, order1)) 
    #assert len(set(order1)) == len(order1), '%d %d' % (len(set(order1)), len(order1))
    return dist1, order1 
    
def do3opt_1(N, distances, dist, order, deltas, boundaries):        
    (p1, p2, p3), (p1a, p2a, p3a) = boundaries
    order1 = order.copy()
    
    # :a e:d c:b f:
    n = p1a 
    order1[n:n + p3-p2a+1] = reverse_order(order, p2a, p3)
    n += p3-p2a+1
    order1[n:n + p2-p1a+1] = order[p1a:p2+1]
    n += p2-p1a+1
    #print n, N
    if n < N:
        order1[n:N] = order[p3a:]
    
    if DEBUG:
        print np.arange(N), 'indexes'
        print order, trip2(distances, order)
        print order1, trip2(distances, order1)
    dist1 = dist + deltas[1]
    show_path2(dist, order, dist1, order1, boundaries)
    #assert CLOSE(dist1, trip2(distances, order1)), DIFF(dist1, trip2(distances, order1)) 
    #assert len(set(order1)) == len(order1), '%d %d' % (len(set(order1)), len(order1))
    return dist1, order1 
    
def do3opt_2(N, distances, dist, order, deltas, boundaries):       
    (p1, p2, p3), (p1a, p2a, p3a) = boundaries
    order1 = order.copy()
    #print 'do3opt_2:', N, (p1, p2, p3), (p1a, p2a, p3a)
    
    # :a c:b e:d f:
    #print ':%d %d:%d %d:%d %d:' % (p1, p2, p1a, p3, p2a, p3a)
    n = p1a 
    order1[n:n + p2-p1a+1] = reverse_order(order, p1a, p2) 
    n += p2-p1a+1
    order1[n:n + p3-p2a+1] = reverse_order(order, p2a, p3)
    n += p3-p2a+1
    if n < N:
        order1[n:N] = order[p3a:]
    
    if DEBUG:
        print np.arange(N), 'indexes'
        print order, trip2(distances, order)
        print order1, trip2(distances, order1)
    dist1 = dist + deltas[2]
    show_path2(dist, order, dist1, order1, boundaries)
    #assert CLOSE(dist1, trip2(distances, order1)), DIFF(dist1, trip2(distances, order1)) 
    #assert len(set(order1)) == len(order1), '%d %d' % (len(set(order1)), len(order1))
    return dist1, order1 

def do3opt_3(N, distances, dist, order, deltas, boundaries):        
    (p1, p2, p3), (p1a, p2a, p3a) = boundaries
    order1 = order.copy()
    
    # :a d:e b:c f:
    n = p1a 
    order1[n:n + p3-p2a+1] = order[p2a:p3+1]
    n += p3-p2a+1
    order1[n:n + p2-p1a+1] = order[p1a:p2+1]
    n += p2-p1a+1
    if n < N:
        order1[n:N] = order[p3a:]
    
    if DEBUG:
        print np.arange(N), 'indexes'
        print order, trip2(distances, order)
        print order1, trip2(distances, order1)
    dist1 = dist + deltas[3]
    show_path2(dist, order, dist1, order1, boundaries)
    #assert CLOSE(dist1, trip2(distances, order1)), DIFF(dist1, trip2(distances, order1)) 
    #assert len(set(order1)) == len(order1), '%d %d' % (len(set(order1)), len(order1))
    return dist1, order1  
    
do3_all = [do3opt_0, do3opt_1, do3opt_2, do3opt_3]    
    
def do3opt_any(N, distances, dist, order, selection):
    """
        a p1
        b p1a
        c p2
        d p2a
        e p3
        f p3a
    """
    
    assert CLOSE(dist, trip2(distances, order)), '%s %s' % (dist, trip2(distances, order))
    
    deltas, boundaries = calc3opt(N, distances, order, dist)
    assert all((dist + d > 0) for d in deltas) 
    
    return ([dist + d for d in deltas], 
            [do3(N, distances, dist, order, deltas, boundaries) 
                for do3 in do3_all]) 

    
def do3opt_best(N, distances, dist, order, max_iter):
    
    assert len(set(order)) == len(order)
    for _ in xrange(max_iter):
        deltas, boundaries = calc3opt(N, distances, order, dist)
        #print 'deltas:', deltas
        if any(d < 0 for d in deltas):
            break
    #print 'best deltas:', deltas        
    imin = min(list(enumerate(deltas)), key=lambda x: x[1])[0]        
    # Super inefficient !@#$
    #print 'imin:', type(imin), imin
    dist1, order1 = do3_all[imin](N, distances, dist, order, deltas, boundaries)
    return dist1, order1

MAX_CLOSEST = 20
def do3opt_local(N, distances, closest, dist, order):
    
    #assert len(set(order)) == len(order)
    best = (0.0, None, None)
    M = min(N, MAX_CLOSEST)
    
    #print 'N M', N, M
    for p1 in xrange(N - 4):
        # for p2 in xrange(p1+2, N - 2):
        n2 = 0
        for p2 in closest[p1]:
            if p2 < p1 + 2: continue
            if n2 >= M: break
            n2 += 1
            boundary_starts = (p1, p2)
            delta, boundaries = calc2opt_delta(N, distances, order, dist, boundary_starts)
            if delta < best[0] - EPSILON:
                best = delta, boundaries
            #print (n2-1, p2),     
    #print 'n2:', n2            
    
    for p1 in xrange(N - 6):
        #for p2 in xrange(p1+2, N - 4):
        n2 = 0
        for p2 in closest[p1]:
            if p2 < p1 + 2: continue
            if n2 >= M: break
            n2 += 1
            #for p3 in xrange(p2+2, N - 2):
            n3_1 = 0
            done_starts = set()
            for p3 in closest[p1]:
                if p3 < p2 + 2: continue
                if n3_1 >= M: break
                
                boundary_starts = (p1, p2, p3)
                done_starts.add(boundary_starts)
                n3_1 += 1
                
                deltas, boundaries = calc3opt_deltas(N, distances, order, dist, boundary_starts)
                for d in deltas:
                    if d < best[0] - EPSILON:
                        best = d, deltas, boundaries
                #print (n3_1-1, p3),        
            n3_2 = 0
            for p3 in closest[p2]:
                if p3 < p2 + 2: continue
                if n3_2 >= M: break
                boundary_starts = (p1, p2, p3)
                if boundary_starts in done_starts: continue
                done_starts.add(boundary_starts)
                n3_2 += 1
                
                deltas, boundaries = calc3opt_deltas(N, distances, order, dist, boundary_starts)
                for d in deltas:
                    if d < best[0] - EPSILON:
                        best = d, deltas, boundaries            
    #print 'n23:', n2, n3_1, n3_2 
    
    dist1, order1 = dist, order
    #print 'best:', best                    
    if best[0] < 0.0:
        if len(best) == 2: # 2-opt
            delta, boundaries = best
            #print '$', boundaries
            (p1, p2), (p1a, p2a) = boundaries
            dist1, order1 = do2opt(N, distances, dist, order, delta, p1, p2, p1a, p2a)
        elif len(best) == 3: # 3-opt    
            d, deltas, boundaries = best
            #print 'best deltas:', deltas
            #imin = min(list(enumerate(deltas)), key=lambda x: x[1])[0] 
            imin = 0
            for i in (1, 2, 3):
                if deltas[i] < deltas[imin]:
                    imin = i
                   
            dist1, order1 = do3_all[imin](N, distances, dist, order, deltas, boundaries)
        
    return dist1, order1 

def local_search(N, distances, closest, dist, order):
    
    changed = False
    while True:
        dist1, order1 = do3opt_local(N, distances, closest, dist, order)    
        assert dist1 <= dist
        if dist1 == dist:
            break
        dist, order = dist1, order1 
        changed = True
    if changed:
        normalize(N, order)
    return dist, order
    
def search(N, distances, visited, hash_base, dist, order):
    """Search for best solution starting with order"""
  
    #assert isinstance(dist, float), type(dist)
    #assert isinstance(order, np.ndarray), type(order)
    #assert CLOSE(dist, trip2(distances, order))
    #print '@1', order.shape
  
    MAX_NO_IMPROVEMENT = 20 
    MAX_ITER = 40 
    MAX_NEIGHBORHOOD = 30 
    
    local_min_count = 0
    
    best = (dist, order)
    
    dist0 = dist    
    #print 'search', dist
    
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
            #dist, order = do3opt_best(N, distances, dist, order, MAX_ITER)
            #dist, order = do3opt_local(N, distances, dist, order)
            dist, order = local_search(N, distances, closest, dist, order)
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
            #if i % 1000 == 100:
                #print '$$', neighborhood, no_improvement_count, i
        else: # increment the count as we did not find a local optima
            no_improvement_count +=1        
                
        #print '**', neighborhood, no_improvement_count, i, best[0]   
        visited.add(hsh)      

    print 'Done search', i, dist0, best    
    return best                
    
NUM_SOLUTIONS = 40 
MAX_ITER = 100 
        
def solve(points):
    """Return traversal order of points that minimizes distance travelled"""
    
    N, locations, distances, closest = precalculate(points)
        
    hash_base = np.random.randint(10**4, 10**6, N)
    visited = set()
    
    outer_solutions = []
    optimum_solutions = []
    
    for start in xrange(N):
        print '$%d' % start,
        dist, order = populate_greedy(N, distances, closest, start)
        #if start % 3 > 0:
        #    random.shuffle(order)
        #    dist = trip2(distances, order)
        
        normalize(N, order)
         
        assert len(set(order)) == len(order), start
        assert CLOSE(dist, trip2(distances, order)), '%s %s' % (dist, trip2(distances, order))
        
        hsh = np.dot(hash_base, order)   
        if hsh in visited:   # Done this local search?
            continue
        visited.add(hsh)     
        dist, order = local_search(N, distances, closest, dist, order)
        assert CLOSE(dist, trip2(distances, order)), '%s %s' % (dist, trip2(distances, order))
        assert dist > 0
                       
        assert len(set(order)) == len(order), start
        hsh = np.dot(hash_base, order)
        #print hash_base.shape, order.shape, hsh.shape
        #assert hsh not in visited
        visited.add(hsh)    
        outer_solutions.append((dist, hsh, order))
        if not optimum_solutions or dist < optimum_solutions[-1][0]:
            optimum_solutions.append((dist, hsh, order))
            print 'best:', optimum_solutions[-1][0]
            
    print 'Done greedy', len(outer_solutions), len(optimum_solutions)
    
    for dist, hsh, order in outer_solutions:
        
        hsh = np.dot(hash_base, order)   
        if hsh in visited:   # Done this local search?
            continue
        dist, order = search(N, distances, visited, hash_base, dist, order)
        assert dist > 0
        if dist < optimum_solutions[-1][0]:
            optimum_solutions.append((dist, hsh, order))
            print 'best:', optimum_solutions[-1][0]
    
    dist, _, order = optimum_solutions[-1]
    
    print 'optimum:', dist, len(optimum_solutions), [x[0] for x in optimum_solutions[-10:]]
    
    return dist, list(order)
    
    
def solveIt(inputData, path=None):
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
    
    if path:
        save_solution(path, points, dist, order)
        
    # prepare the solution in the specified output format
    outputData = str(dist) + ' ' + str(0) + '\n'
    outputData += ' '.join(map(str, order))

    return outputData

def loadInputData(fileLocation):
    inputDataFile = open(fileLocation, 'r')
    inputData = ''.join(inputDataFile.readlines())
    inputDataFile.close()
    return inputData   
   
fileNameLookup = {'WdrlJtJq': './data/tsp_51_1',
 'dTewhF6o': './data/tsp_100_3',
 'OlJBvw72': './data/tsp_200_2',
 'KZQYdfck': './data/tsp_574_1',
 'vViyejW4': './data/tsp_1889_1',
 'vLKzhJhP': './data/tsp_33810_1',
 'WdrlJtJq-dev': './data/tsp_51_1',
 'dTewhF6o-dev': './data/tsp_100_3',
 'OlJBvw72-dev': './data/tsp_200_2',
 'KZQYdfck-dev': './data/tsp_574_1',
 'vViyejW4-dev': './data/tsp_1889_1',
 'vLKzhJhP-dev': './data/tsp_33810_1'} 
 
partIds = ['WdrlJtJq',
 'dTewhF6o',
 'OlJBvw72',
 'KZQYdfck',
 'vViyejW4',
 'vLKzhJhP'] 

path_list = [fileNameLookup[id] for id in partIds]

for path in path_list:
    print '-' * 80
    print path
    solution = solveIt(loadInputData(path), path)
    print solution
    

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

