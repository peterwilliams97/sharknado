#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
TODO: Lower M for 3-opt
      Save more often
      M=10 for 33000 instance
"""
from __future__ import division
import math, random
import numpy as np
from itertools import count

import sys, time, os

from numba import autojit, jit, double

import best_history

# 250 forwards, 251 backwards
VERSION = 328

MAX_CLOSEST = 100 # 2000
MAX_N = 30 * 1000
DEBUG = False
EPSILON = 1e-6
RANDOM_SEED = 208 # Not the Nelson!
MAX_EDGES = 100 # 2000

print 'VERSION=%d' % VERSION
print 'MAX_CLOSEST=%d' % MAX_CLOSEST
print 'MAX_N=%d' % MAX_N
print 'MAX_EDGES=%d' % MAX_EDGES

random.seed(RANDOM_SEED)


def CLOSE(a, b):
    return abs(a - b) < 10.0
 
def DIFF(a, b): 
    return 'DIFF: %s - %s = %s' % (a, b, a - b)    

random.seed(111)

@autojit
def length(point1, point2):
    return np.hypot(point1, point2)
    #return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


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
        if i % 1000 == 999:
            print 'i=%d,nodes=%d,%d' % (i, len(all_nodes - nodes), len(nodes))
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
    

def load_save_npa(base_dir, npa_dict, do_save):
    if do_save:
        try:
            os.makedirs(base_dir)
        except:
            pass
    else:
        for name in npa_dict:
            path = os.path.join(base_dir, name + '.npy')
            if not os.path.exists(path):
                print '%s does not exist' % path
                return False
            
    for name in npa_dict:
        path = os.path.join(base_dir, name + '.npy')
        if do_save: 
            print 'saving', path, npa_dict[name].shape, 
            np.save(path, npa_dict[name])
            print '!'
        else:
            print 'loading', path
            npa_dict[name] = np.load(path, mmap_mode='r')
            print npa_dict[name].shape, '!' 
            
    return True        

    
CACHE_DIR = 'cache'    
#hack_locations = None 
  
def precalculate(points): 
    #global hack_locations
    N = len(points)
    print '@@1'
    
    base_dir = os.path.join(CACHE_DIR, 'obs%05d' % N)
    npa_dict = { 'locations': None,   'distances': None, 'closest': None,  }
    print '@@2', base_dir
    
    existing = load_save_npa(base_dir, npa_dict, False)
    if existing:
        locations, distances, closest = npa_dict['locations'], npa_dict['distances'], npa_dict['closest'] 
        print 'loading existing !!!!!!!!!!!!!!!!!!'
        print 'locations:', locations.shape
        print 'distances:', distances.shape
        print 'closest:', closest.shape

     
        assert locations.shape[0] == N
        assert distances.shape[0] == N
        assert distances.shape[1] == N
        assert closest.shape[0] == N
        assert closest.shape[1] == N - 1
        #for i in xrange(1, N):
        #    for j in xrange(i):
        #        assert distances[i, j] > 0.0
    else:
    
        locations = np.array(points, dtype=np.float64)
        distances = np.empty((N, N), dtype=np.float64)  # !@#$%
        closest = np.empty((N, N-1), dtype=np.int32)
        row = np.empty(N, dtype=np.float64)
        
        print '@@2a'
        print 'locations:', locations.shape
        print 'distances:', distances.shape
        print 'closest:', closest.shape
        
        if True: # !@#$
            start_time = time.time()
            for i in xrange(N):
                if i % 1000 == 100:
                    dt = max(0.1, time.time() - start_time)
                    print (i, i/N, dt, i/dt, (N - i)/(i/dt)/3600.0)
                diff = locations - locations[i]
                for j in xrange(N):
                    row[j] = np.sqrt(diff[j][0] ** 2 + diff[j][1] ** 2)
                distances[i,:] = row[:] 
                # Ordering of distances, closest first.
                a = range(N)
                a.sort(key=lambda j: distances[i,j])
                closest[i, :] = a[1:]     
 
            print '@@4'    
        
        npa_dict['locations'], npa_dict['distances'], npa_dict['closest'] = locations, distances, closest
        load_save_npa(base_dir, npa_dict, True)
    

    print '@@5'    
    #hack_locations = locations    
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
saved_paths = set() 
saved_points = {}    
saved_solutions = {}    
def save_solution(path, points, dist, order):
    saved_paths.add(path)
    saved_points[path] = points
    saved_solutions[path] = (dist, list(order)) 
    saved_scores = { path: soln[0] for path, soln in saved_solutions.items() }
    history = HISTORY % VERSION
    print 'Writing history:', history, sys.argv[0]
    with open(history, 'wt') as f:
        f.write('from numpy import array\n\n')
        f.write('prog="%s"\n' % sys.argv[0])
        f.write('VERSION=%d\n' % VERSION)
        f.write('MAX_CLOSEST=%d\n' % MAX_CLOSEST)
        f.write('MAX_N=%d\n' % MAX_N)
        f.write('MAX_EDGES=%d\n' % MAX_EDGES)
        f.write('RANDOM_SEED=%d\n' % RANDOM_SEED)
        f.write('DEBUG=%s\n' % DEBUG)
        f.write('EPSILON=%s\n' % EPSILON)
        f.write('saved_paths = %s\n' % repr(sorted(saved_paths)))
        f.write('saved_scores = %s\n' % repr(saved_scores))
        s1 = repr(saved_solutions)
        assert '...' not in s1, s1
        s2 = repr(saved_points)
        assert '...' not in s2, s2
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
    order1 = order.copy() # make a copy
    order1[p1a:p2+1] = reverse_order(order, p1a, p2)
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


def do3opt_0(N, order, boundaries):        
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
    
    return order1 
    
def do3opt_1(N, order, boundaries):        
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

    return  order1 
    
def do3opt_2(N, order,  boundaries):       
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

    return  order1 

def do3opt_3(N, order,  boundaries):        
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
        
    return order1  
    
do3_all = [do3opt_0, do3opt_1, do3opt_2, do3opt_3]    
    

"""
    a p1
    b p1a
    c p2
    d p2a
    e p3
    f p3a
"""
    

    
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




#from numba.decorators import jit
#nd4type = numba.double[:,:,:,:]

import numba

#@jit(argtypes=(numba.int32, numba.float_[:,:], numba.float_[:,:], numba.int32[:]))
#@autojit
def find_2_3opt_min(N, distances, closest, order, dist, do3):
    
    N1 = N - 1
    N2 = N - 2
    N4 = N - 4
    M2 = min(N1, MAX_CLOSEST)
    M3 = int(min(N1, math.sqrt(MAX_CLOSEST)))
        
    delta_ = 0.0
    p1_, p2_, p3_ = -1, -1, -1
    opt3_i = -1
    opt3deltas = np.zeros(4)
     
    def make_closest_order(m):    
        closest_order = range(m) 
        n = 1
        while n < N4:
            n = min(10 * n, N4)
            if n <= len(closest_order): continue  
            new_order = range(len(closest_order), n)
            random.shuffle(new_order)
            closest_order.extend(new_order[:m])
        m_new = len(closest_order)
        existing = set(closest_order)
        for i in range(N-1):
            if i not in existing:
                closest_order.append(i)
                
        return m_new, closest_order

    #print 'N, closest', N, closest.shape
    #print 'M2, M3, before:', M2, M3
    M2, closest_order2 = make_closest_order(M2)
    M3, closest_order3 = make_closest_order(M3)  

    #print 'M2, M3, after:', M2, M3
    #print 'closest_order2, closest_order3:', closest_order2[-1], closest_order3[-1] 
    assert M2 <= closest.shape[1], M2
    assert M3 <= closest.shape[1], M3
        
    counter = count()
    
    for p1 in xrange(N - 4):
        # for p2 in xrange(p1+2, N - 2):
        #n2 = 0
        n2cnt = 0
        closest1 = closest[p1]
        for n2 in xrange(N2):
            cnt = next(counter)
            if cnt % 1000000 == 100000:  print 'cnt2=%d,(p1=%d,p2=%d),n2=%d,dist=%.1f,delta_=%.1f' % (cnt, p1, p2, n2, dist+delta_, delta_)
           
            #print n2, 
            #print closest_order2[n2],
            #print closest1[closest_order2[n2]]
            p2 = closest1[closest_order2[n2]]
            #n2 += 1
            if p2 < p1 + 2 or p2 > N2: continue
                            
            w1, w2 = order[p1], order[p1 + 1]   # a b
            w3, w4 = order[p2], order[p2 + 1]   # c d 
            delta = (distances[w1, w3] + distances[w2, w4]) - (distances[w1, w2] + distances[w3, w4])
                        
            if delta < delta_:
                delta_ = delta
                p1_, p2_ = p1, p2
                
            n2cnt += 1
            if n2cnt > M2: break 
       
    if do3:
        counter2 = count()
        done_p3 = set()     
        for p1 in xrange(N - 6):
            #for p2 in xrange(p1+2, N - 4):
            #n2 = 0
            #for p2 in closest[p1]:
            #    if p2 < p1 + 2: continue
            #    if n2 >= M2: break
            n2cnt = 0
            closest1 = closest[closest_order3[p1]]
            for n2 in xrange(N1):
                #cnt = next(counter2)
                #if cnt % 1000000 == 500:
                #    print '**cnt=%d,p1=%d,n2=%d,n3=%d' % (cnt, p1, n2, n3) 
            
                p2 = closest1[n2]
                if p2 < p1 + 2 or p2 > N - 4: continue 
                            
                #for p3 in xrange(p2+2, N - 2):
                n3cnt = 0
                closest2 = closest[p2]
                
                for n3 in xrange(N1):
                    cnt = next(counter2)
                    if cnt % 1000000 == 100000:
                        print 'cnt3=%d,(p1=%d,p2=%d,p3=%d),n2=%d,n3=%d,dist=%.1f,delta_=%.1f' % (cnt, p1, p2, p3, n2, n3, dist+delta_, delta_)
                        
                    p3_1 = closest1[n3]
                    p3_2 = closest2[n3]
                    #n3 += 1
                    
                    #p3_all = []
                    #if p3_1 >= p2 + 2 and p3_1 < N -2: p3_all.append(p3_1)
                    #if p3_2 >= p2 + 2 and p3_2 < N -2: p3_all.append(p3_2)
                    
                    for p3 in (p3_1, p3_2):
                        if not (p3 >= p2 + 2 and p3 < N2): continue
                        if (p1,p2,p3) in done_p3: continue
                        done_p3.add((p1,p2,p3))
                                              
                        w1, w2 = order[p1], order[p1+1]   # a b
                        w3, w4 = order[p2], order[p2+1]   # c d  
                        w5, w6 = order[p3], order[p3+1]   # e f  
                        
                        bf = distances[w1, w2] + distances[w3, w4] + distances[w5, w6] # Original distance 
                        opt3deltas[0] = distances[w1, w4] + distances[w2, w6] + distances[w3, w5] - bf
                        opt3deltas[1] = distances[w1, w5] + distances[w2, w4] + distances[w3, w6] - bf
                        opt3deltas[2] = distances[w1, w3] + distances[w2, w5] + distances[w4, w6] - bf
                        opt3deltas[3] = distances[w1, w4] + distances[w2, w5] + distances[w3, w6] - bf 
                       
                        for i in xrange(4):
                            if opt3deltas[i] < delta_:
                                delta_ = opt3deltas[i]
                                opt3_i = i
                                p1_, p2_, p3_ = p1, p2, p3
                        
                        n3cnt += 1
                    if n3cnt > M3: 
                        #print (n3cnt,),
                        break 
                    
                n2cnt += 1
                
                if n2cnt > M3: break
            #print ('*', n2cnt, n2, cnt)
            
    return delta_, p1_, p2_, p3_, opt3_i                              
    #return delta_, np.array([p1_, p2_, p3_, opt3_i])                   

#@autojit
def get_crossed_edges(N, locations, closest, order, max_edges):
       
    print 'get_crossed_edges: N=%d,max_edges=%d' % (N, max_edges)
    max_edges = min(N-2, max_edges)
    print 'max_edges=%d' % (max_edges)
    
    crossed_edges = []
    
    #def plot_line(i0, i1, color):
    #    x1 = [locations[order[i],0] for i in i0, i1]
    #    y1 = [locations[order[i],1] for i in i0, i1]
    #    plt.plot(x1, y1, marker='.', linestyle='-', color=color)
        
    #def plot_pair(i, j):
    #    plot_line(i, i+1, 'r')   
    #    plot_line(j, j+1, 'b') 
    
    for i in xrange(2, N-1):
    
        if i % 100 == 99: print i,
        ei0 = locations[order[i],:] 
        ei1 = locations[order[i+1],:]
        
        xi_min = min(ei0[0], ei1[0])
        xi_max = max(ei0[0], ei1[0])
        yi_min = min(ei0[1], ei1[1])
        yi_max = max(ei0[1], ei1[1])
        
        # y = ax + b : i
        i_vertical = abs(ei1[0] - ei0[0]) < EPSILON
        if not i_vertical:
            ai = (ei1[1] - ei0[1])/(ei1[0] - ei0[0])  
            bi = ei0[1] - ai * ei0[0] 
            #assert abs(bi - (ei1[1] - ai * ei1[0])) < EPSILON 
        
        closest1 = closest[i,:]
        
        for n in xrange(max_edges):
            j = closest1[n]
            if j > i-2:
                continue
            ej0 = locations[order[j],:] 
            ej1 = locations[order[j+1],:] 
                                    
            xj_min = min(ej0[0], ej1[0])
            xj_max = max(ej0[0], ej1[0])
            yj_min = min(ej0[1], ej1[1])
            yj_max = max(ej0[1], ej1[1])
            
            if xj_min > xi_max or xj_max < xi_min:
                continue
            if yj_min > yi_max or yj_max < yi_min:
                continue    
  
            # y = ax + b : j
            j_vertical = abs(ej1[0] - ej0[0]) < EPSILON
            if not j_vertical:
                aj = (ej1[1] - ej0[1])/(ej1[0] - ej0[0])  
                bj = ej0[1] - aj * ej0[0] 
                #assert abs(bj - (ej1[1] - aj * ej1[0])) < EPSILON
                
            # Intersection of x values
            x_min = max(xi_min, xj_min)
            x_max = min(xi_max, xj_max)
            
            # Handle the vertical cases
            if i_vertical or j_vertical:
                if i_vertical and j_vertical:
                    if xj_min == xi_min:
                        crossed_edges.append((j, i))
                        continue

            # yi at min_x (left of intersection)
            if i_vertical:
                yi_min_x = yi_min 
                yi_max_x = yi_max
            else:    
                yi_min_x = ai * x_min + bi 
                yi_max_x = ai * x_max + bi 
                
            # yj at min_x (left of intersection)
            if j_vertical:
                yj_min_x = yj_min 
                yj_max_x = yj_max
            else: 
                yj_min_x = aj * x_min + bj 
                yj_max_x = aj * x_max + bj 

            # Crossed?
            if ( (yi_min_x >= yj_min_x and yi_max_x <= yj_max_x) 
              or (yi_min_x <= yj_min_x and yi_max_x >= yj_max_x)):
                
                #print ei0, ei1 
                #print ej0, ej1
                #exit() 
                crossed_edges.append((j, i))
                
                
                #    ei0 = np.array(locations[order[i],:]) 
                #    ei1 = np.array(locations[order[i+1],:])
                #    ej0 = np.array(locations[order[j],:]) 
                #    ej1 = np.array(locations[order[j+1],:]) 
                #    print '~~ %d %d : %s %s' % (i, j, (ei0, ei1), (ej0, ej1))
                #    print ai, bi
                #    print aj, bj
                #    print x_min, yi_min_x, yj_min_x
                #    print x_max, yi_max_x, yj_max_x
                #    plot_pair(i, j)
                #    plt.show() 

    
    for ij in crossed_edges:
        i = ij[0]
        j = ij[1]
        ei0 = np.array(locations[order[i],:]) 
        ei1 = np.array(locations[order[i+1],:])
        ej0 = np.array(locations[order[j],:]) 
        ej1 = np.array(locations[order[j+1],:]) 
        print '~~ %d %d : %s %s' % (i, j, (ei0, ei1), (ej0, ej1))
        #plot_pair(i, j)
    #plt.show()    
    #exit()    
    print 'found %d crossed edges' % len(crossed_edges)
    
    return crossed_edges        
    

def remove_crossed_edges(N, locations, distances, closest, order, dist, max_edges):
    
    crossed_edges = get_crossed_edges(N, locations, closest, order, max_edges)
            
    N2 = N - 2
    
    cross_vertices = set()
    for p1, p2 in crossed_edges:
        cross_vertices.add(p1)
        cross_vertices.add(p2)
    cross_vertices = sorted(cross_vertices)    
            
    for i, p1 in enumerate(cross_vertices):
        delta_ = 0.0
        p2_ = -1
        closest1 = closest[p1]
        for n2 in xrange(N - 1):
            p2 = closest1[n2]
            if abs(p1 - p2) < 2 or p2 >= N2: continue
                            
            w1, w2 = order[p1], order[p1 + 1]   # a b
            w3, w4 = order[p2], order[p2 + 1]   # c d 
            delta = (distances[w1, w3] + distances[w2, w4]) - (distances[w1, w2] + distances[w3, w4])
                        
            if delta < delta_:
                delta_ = delta
                p2_ = p2
                
        if delta_ < 0.0:
           
            assert p1 >= 0, p1
            assert p2_ >= 0, p2_
            if p2_ < p1:
                p1, p2_ = p2_, p1
            dist, order = do2opt(N, distances, dist, order, delta_, p1, p2_, p1+1, p2_+1)
            print '@i=%d: p1=%d,p2=%d,dist=%.1f,delta=%.1f' % (i, p1, p2, dist, delta_)
       
    
    return dist, order, len(crossed_edges)    
       
    
    
def find_2_3opt_long_edges(N, distances, closest, order, dist, do3):
    
    #do3 = True
    edge_lengths = [distances[order[i], order[i+1]] for i in xrange(N-1)]
    edge_lengths.append(distances[order[N-1], order[0]])
    edges = range(N)
    edges.sort(key=lambda i: -edge_lengths[i])
     
    N_EDGES = int(min(10.0*math.sqrt(N), N))
    #N_EDGES = int(min(math.sqrt(N), N))
    edges = edges[:N_EDGES]
    
    #print 'N_EDGES=%d' % N_EDGES
        
    N1 = N - 1
    N2 = N - 2
    #N4 = N - 4
    M2 = min(N1, MAX_CLOSEST)
    M3 = int(min(N1, math.sqrt(MAX_CLOSEST)))
    N_EDGES_M3 = min(N_EDGES, M3)
        
    delta_ = 0.0
    p1_, p2_, p3_ = -1, -1, -1
    opt3_i = -1
    opt3deltas = np.zeros(4)
        
    counter = count()
    
    for p1 in edges:
        n2cnt = 0
        closest1 = closest[p1]
        for n2 in xrange(N2):
            cnt = next(counter)
            if cnt % 1000000 == 100000:
                print '_cnt2=%d,(p1=%d,p2=%d),n2=%d,dist=%.1f,delta_=%.1f' % (cnt, p1, p2, n2, dist+delta_, delta_)
           
            p2 = closest1[n2]
            #n2 += 1
            if p2 < p1 + 2 or p2 > N2: continue
                            
            w1, w2 = order[p1], order[p1 + 1]   # a b
            w3, w4 = order[p2], order[p2 + 1]   # c d 
            delta = (distances[w1, w3] + distances[w2, w4]) - (distances[w1, w2] + distances[w3, w4])
                        
            if delta < delta_:
                delta_ = delta
                p1_, p2_ = p1, p2
                
            n2cnt += 1
            #if n2cnt > M2: break 
       
    if do3:
        counter2 = count()
        done_p3 = set()     
        for p1 in edges:

            n2cnt = 0
            closest1 = closest[p1]
            for n2 in xrange(N1):
                #cnt = next(counter2)
                #if cnt % 1000000 == 500:
                #    print '**cnt=%d,p1=%d,n2=%d,n3=%d' % (cnt, p1, n2, n3) 
            
                p2 = closest1[n2]
                if p2 < p1 + 2 or p2 > N - 4: continue 
                            
                #for p3 in xrange(p2+2, N - 2):
                n3cnt = 0
                closest2 = closest[p2]
                
                for n3 in xrange(N1):
                    cnt = next(counter2)
                    if cnt % 1000000 == 100000:
                        print '_cnt3=%d,(p1=%d,p2=%d,p3=%d),n2=%d,n3=%d,dist=%.1f,delta_=%.1f' % (cnt, p1, p2, p3, n2, n3, dist+delta_, delta_)
                        
                    p3_1 = closest1[n3]
                    p3_2 = closest2[n3]
                    
                    for p3 in (p3_1, p3_2):
                        if not (p3 >= p2 + 2 and p3 < N2): continue
                        if (p1,p2,p3) in done_p3: continue
                        done_p3.add((p1,p2,p3))
                                              
                        w1, w2 = order[p1], order[p1+1]   # a b
                        w3, w4 = order[p2], order[p2+1]   # c d  
                        w5, w6 = order[p3], order[p3+1]   # e f  
                        
                        bf = distances[w1, w2] + distances[w3, w4] + distances[w5, w6] # Original distance 
                        opt3deltas[0] = distances[w1, w4] + distances[w2, w6] + distances[w3, w5] - bf
                        opt3deltas[1] = distances[w1, w5] + distances[w2, w4] + distances[w3, w6] - bf
                        opt3deltas[2] = distances[w1, w3] + distances[w2, w5] + distances[w4, w6] - bf
                        opt3deltas[3]= distances[w1, w4] + distances[w2, w5] + distances[w3, w6] - bf 
                       
                        for i in xrange(4):
                            if opt3deltas[i] < delta_:
                                delta_ = opt3deltas[i]
                                opt3_i = i
                                p1_, p2_, p3_ = p1, p2, p3
                        
                        n3cnt += 1
                    if n3cnt > N_EDGES_M3: 
                        #print (n3cnt,),
                        break 
                    
                n2cnt += 1
                
                if n2cnt > N_EDGES: break
            #print ('*', n2cnt, n2, cnt)
            
    return delta_, p1_, p2_, p3_, opt3_i                              
    #return delta_, np.array([p1_, p2_, p3_, opt3_i])                   

    
def do3opt_local(N, distances, closest, dist, order, long_edges, do3):
    
    find_min = find_2_3opt_long_edges if long_edges else find_2_3opt_min
    
    #assert len(set(order)) == len(order)
    delta, p1, p2, p3, opt3_i = find_min(N, distances, closest, order, dist, do3)
    
    dist1, order1 = dist, order
    #print 'best:', best 

    if delta < 0.0:
        if p3 < 0: # 2-opt
            dist1, order1 = do2opt(N, distances, dist, order, delta, p1, p2, p1+1, p2+1)
        else: # 3-opt    
            dist1 = dist + delta
            order1 = do3_all[opt3_i](N, order, ((p1, p2, p3), (p1+1, p2+1, p3+1)))
        
    return dist1, order1 

def local_search(N, distances, closest, dist, order, long_edges, do3):
    
    changed = False
    while True:
        dist1, order1 = do3opt_local(N, distances, closest, dist, order, long_edges, do3)    
        assert dist1 <= dist
        if dist1 == dist:
            break
        dist, order = dist1, order1 
        changed = True
    if changed:
        normalize(N, order)
    return dist, order
    
def do2opt_any(N, distances, dist, order):
    
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
        
    p2b = p2 - 1 if p2 > 0 else N1 
    p2a = p2 + 1 if p2 < N1 else 0    
        
    # to ensure we always have p1<p2        
    if p2 < p1:
        p1, p2 = p2, p1
        p1a, p2a = p2a, p1a


    delta, boundaries = calc2opt_delta(N, distances, order, dist, (p1, p2))
    #assert dist + delta > 0
    
    #print '%', boundaries
    return do2opt(N, distances, dist, order, delta, p1, p2, p1a, p2a)
    

def do2opt_random(N, distances, dist, order):
    
    #assert CLOSE(dist, trip2(distances, order)), '%s %s' % (dist, trip2(distances, order))
    
    done = set()
    
    print 'do2opt_random'
    MAX_R = (N * N)/4
    while len(done) < MAX_R:
        p1, p2 = random.randrange(0, N), random.randrange(0, N)
        # do this so as not to overshoot tour boundaries
     
        N1 = N - 1
        p1b = p1 - 1 if p1 > 0 else N1 
        p1a = p1 + 1 if p1 < N1 else 0
        w0, w1, w2 = order[p1b], order[p1], order[p1a]
            
        exclude = set([w0, w1, w2])
         
        while order[p2] in exclude:
            p2 = random.randrange(0, N)
            
        p2b = p2 - 1 if p2 > 0 else N1 
        p2a = p2 + 1 if p2 < N1 else 0    
            
        # to ensure we always have p1<p2        
        if p2 < p1:
            p1, p2 = p2, p1
            p1a, p2a = p2a, p1a
            
        done.add((p1,p2))    
        
        delta, boundaries = calc2opt_delta(N, distances, order, dist, (p1, p2))
        if delta < 0:
           break
    #assert dist + delta > 0
    
    print N, (MAX_R, len(done)), delta 
    #print '%', boundaries
    
    if delta < 0:
        return do2opt(N, distances, dist, order, _delta, p1, p2, p1a, p2a)
    else:
        return dist, order
        
def do2opt_random(N, distances, dist, order):
    
    #assert CLOSE(dist, trip2(distances, order)), '%s %s' % (dist, trip2(distances, order))
    
    #assert len(set(order)) == len(order)
    
    done = set()
    delta_ = 1e6
    
    N_D = 5000
    N_D2 = N_D//2
    if N > N_D:
        NR0 = random.randrange(0, N - N_D)
        NR1 = NR0 + N_D
    else:
        NR0 = 0
        NR1 = N
        
    print 'NR0, NR1', NR0, NR1    
    
    MAX_R = int(min((N * N)/4, 1e7))
    print 'do2opt_random!', MAX_R, 1e7, (N * N)/4,  (N * N)/4/MAX_R
    for cnt in xrange(MAX_R):
        p1, p2 = random.randrange(NR0, NR1), random.randrange(NR0, NR1)
        # do this so as not to overshoot tour boundaries
     
        N1 = N - 1
        p1b = p1 - 1 if p1 > 0 else N1 
        p1a = p1 + 1 if p1 < N1 else 0
        w0, w1, w2 = order[p1b], order[p1], order[p1a]
            
        exclude = set([w0, w1, w2])
         
        while order[p2] in exclude or abs(order[p2] - w1) > N_D2:
            p2 = random.randrange(0, N)
            
        #assert p1 != p2, '(%d, %d) (%d, %d) %s' % (p1, p2, order[p1], order[p2], exclude)    
            
        p2b = p2 - 1 if p2 > 0 else N1 
        p2a = p2 + 1 if p2 < N1 else 0    
            
        # to ensure we always have p1<p2        
        if p2 < p1:
            p1, p2 = p2, p1
            p1a, p2a = p2a, p1a
  
        if (cnt % 1000000) == 100000: 
            print int(MAX_R), cnt,  len(done), len(done)/MAX_R, cnt/MAX_R     
            
        if (p1,p2) in done:
            continue
        done.add((p1,p2))    
        
        delta, boundaries = calc2opt_delta(N, distances, order, dist, (p1, p2))
        if delta < delta_:
            delta_ = delta
        if delta_ < 0:
           break
    #assert dist + delta > 0
    
    print '$', N, (MAX_R, len(done)), delta_, p1, p2 
    #print '%', boundaries
    
    if delta_ < 0:
        return do2opt(N, distances, dist, order, delta_, p1, p2, p1a, p2a)
    else:
        return dist, order 

def do3opt_random(N, distances, dist, order):
    
    #assert CLOSE(dist, trip2(distances, order)), '%s %s' % (dist, trip2(distances, order))
    
    done = set()
    delta_ = 0
    
    opt3deltas = [0] * 4
    opt3_i = -1
    
    MAX_R = int(min((N * N * N)/4, 1e7))
    print 'do3opt_random!', MAX_R, 1e7, (N * N * N)/4,  (N * N * N)/4/MAX_R
    
    if N > 5000:
        NR0 = random.randrange(0, N -5000)
        NR1 = NR0 + 5000
    else:
        NR0 = 0
        NR1 = N
        
    print 'NR0, NR1', NR0, NR1 
 
    for cnt in xrange(MAX_R):
        p1, p2, p3 = random.randrange(NR0, NR1), random.randrange(NR0, NR1), random.randrange(NR0, NR1)
        # do this so as not to overshoot tour boundaries
     
        N1 = N - 1
        p1b = p1 - 1 if p1 > 0 else N1 
        p1a = p1 + 1 if p1 < N1 else 0
        w0, w1, w2 = order[p1b], order[p1], order[p1a]
            
        exclude = set([w0, w1, w2])
         
        while order[p2] in exclude:
            p2 = random.randrange(NR0, NR1)
            
        p2b = p2 - 1 if p2 > 0 else N1 
        p2a = p2 + 1 if p2 < N1 else 0    
        
        w0, w1, w2 = order[p2b], order[p2], order[p2a]
        exclude = exclude | set([w0, w1, w2]) 
        
        while order[p3] in exclude:
            p3 = random.randrange(NR0, NR1)

        # to ensure we always have p1<p2<p3        
        if p2 < p1: p1, p2 = p2, p1
        if p3 < p1: p1, p3 = p3, p1
        if p3 < p2: p2, p3 = p3, p2
        
        if (cnt % 1000000) == 100000: 
            print int(MAX_R), cnt,  len(done), len(done)/MAX_R, cnt/MAX_R     
  
        
        if (p1,p2,p3) in done: 
            continue     
        done.add((p1,p2,p3)) 
               
        p1a = p1 + 1 if p1 < N1 else 0
        p2a = p2 + 1 if p2 < N1 else 0
        p3a = p3 + 1 if p3 < N1 else 0  
        
        w1, w2 = order[p1], order[p1a]   # a b
        w3, w4 = order[p2], order[p2a]   # c d  
        w5, w6 = order[p3], order[p3a]   # e f  
        
        bf = distances[w1, w2] + distances[w3, w4] + distances[w5, w6] # Original distance 
        opt3deltas[0] = distances[w1, w4] + distances[w2, w6] + distances[w3, w5] - bf
        opt3deltas[1] = distances[w1, w5] + distances[w2, w4] + distances[w3, w6] - bf
        opt3deltas[2] = distances[w1, w3] + distances[w2, w5] + distances[w4, w6] - bf
        opt3deltas[3] = distances[w1, w4] + distances[w2, w5] + distances[w3, w6] - bf 
       
        for i in xrange(4):
            if opt3deltas[i] < delta_:
                delta_ = opt3deltas[i]
                opt3_i = i
                p1_, p2_, p3_ = p1, p2, p3
                p1a_, p2a_, p3a_ = p1a, p2a, p3a
                break

        if delta_ < 0:
           break
    #assert dist + delta > 0
    
    print N, (MAX_R, len(done)), delta_ 
    #print '%', boundaries
 
    if delta_ < 0:
        dist += delta_
        print opt3_i, (p1_, p2_, p3_), (p1a_, p2a_, p3a_)
        order = do3_all[opt3_i](N, order, ((p1_, p2_, p3_), (p1a_, p2a_, p3a_)))
    return dist, order
        
    
def neighbor_search(N, distances, closest, visited, hash_base, dist, order, update_if_necessary):
    """Search for best solution starting with order"""
    
    MAX_NO_IMPROVEMENT_BASE = 2 
    MAX_ITER = 40 
    MAX_NEIGHBORHOOD = 1 
    
    pass1 = False, MAX_NO_IMPROVEMENT_BASE
    pass2 = True, MAX_NO_IMPROVEMENT_BASE//3
    
    best = (dist, order)
    dist0 = dist
    
    
    for do3, MAX_NO_IMPROVEMENT in pass1, pass2:
    
        local_min_count = 0
                    
        no_improvement_count = 0
        counter = count()
        i = next(counter) 
    
        while no_improvement_count <= MAX_NO_IMPROVEMENT:
            print 'no_improvement_count=%d,MAX_NO_IMPROVEMENT=%d' % (no_improvement_count, MAX_NO_IMPROVEMENT)
            # for each neighborhood in neighborhoods
            for neighborhood in range(1, MAX_NEIGHBORHOOD):
                print 'nhd=%d,' % neighborhood,
                
                #Calculate Neighborhood : Involves running stochastic two opt for neighbor times
                for index in range(0, neighborhood):
                    # Get candidate solution from neighborhood
                    dist, order = do2opt_any(N, distances, dist, order)
                    #print '@2', order.shape
                    
                #+print ('@', dist),    
                
                normalize(N, order)            
                hsh = np.dot(hash_base, order)
                #print hash_base.shape, order.shape, hsh.shape
                if hsh in visited:
                    print 'Seen this solution already',
                    continue
                visited.add(hsh)        

                # Refine candidate solution using local search and neighborhood
                #dist, order = do3opt_best(N, distances, dist, order, MAX_ITER)
                #dist, order = do3opt_local(N, distances, dist, order)
                dist, order = local_search(N, distances, closest, dist, order, False, no_improvement_count)
                #if the cost of the candidate is less than cost of current best then replace
                #best with current candidate
                assert dist > 0
                if dist < best[0]:
                    print 'Improvemment %f => %f' % (best[0], dist)
                    update_if_necessary('neighbor_search',  dist, hsh, order)
                    best, no_improvement_count = (dist, order), 0 # We also restart the search when we find the local optima
                    break
                i = next(counter)   
                
            else: # increment the count as we did not find a local optima
                no_improvement_count +=1        
                    
            #print '**', neighborhood, no_improvement_count, i, best[0]   
                 

    print 'Done search', i, dist0 #, best    
    return best                
    
NUM_SOLUTIONS = 40 
MAX_ITER = 100 
        
def solve(path, points):
    """Return traversal order of points that minimizes distance travelled"""
        
    N, locations, distances, closest = precalculate(points)
        
    hash_base = np.random.randint(10**4, 10**6, N)
    visited = set()
    
    outer_solutions = []
    optimum_solutions = []
        
    last_save_time = [time.time()]
    
    def update_if_necessary(title, actual_dist, hsh, order):
        # Save our valuable result before we assert
        if not optimum_solutions or actual_dist < optimum_solutions[-1][0]:
            #actual_dist = trip2(distances, order)
            optimum_solutions.append((actual_dist, hsh, order))
            print 'best %s:' % title, optimum_solutions[-1][0]
            tm = time.time()
            if tm > last_save_time[0] + 60: # !@#$%
                print 'saving:',  tm - last_save_time[0] 
                save_solution(path, points, actual_dist, order)
                last_save_time[0] = tm 
            if not CLOSE(dist, actual_dist):
                print '**********************', DIFF(dist, actual_dist)  
        
    
    dist, order_in = best_history.saved_solutions[path]
    assert len(order_in) == N
    order = np.empty(N, dtype=np.int32)
    for i in xrange(N): 
        order[i] = order_in[i]
             
    hsh = np.dot(hash_base, order)            
    update_if_necessary('best_history', dist, hsh, order)
     
    assert len(set(order)) == len(order), start
    #actual_dist = trip2(distances, order)
    #assert CLOSE(dist, actual_dist), DIFF(dist, actual_dist)
   
    OUTER_N = 100
    LONG_EDGE_N = int(math.sqrt(N))
    print 'OUTER_N:', OUTER_N 
    print 'LONG_EDGE_N:', LONG_EDGE_N
    print 'MAX_EDGES:', MAX_EDGES 
    
    do3 = False
    
    normalize(N, order)
    dist = trip2(distances, order)
    
    assert len(set(order)) == len(order)
    
    #done2 = set()
    #done3 = set()
    while True:
        dist00 = dist
        while True:
            dist0 = dist
            dist, order = do2opt_random(N, distances, dist, order)
            if dist < dist0 - EPSILON:
                dist1 = dist
                normalize(N, order)
                dist = trip2(distances, order)
                assert dist < dist00, '%f %f %f %f (%f) (%f)' % (dist00, dist0, dist, dist1, 
                    dist - dist0, dist - dist1)
                hsh = np.dot(hash_base, order)
                update_if_necessary('do2opt_random', dist, hsh, order)
            else:
                break
            
        dist0 = dist
        dist, order = do3opt_random(N, distances, dist, order)
        if dist < dist0:
            normalize(N, order)
            dist = trip2(distances, order)
            assert dist < dist0 - EPSILON
            hsh = np.dot(hash_base, order)
            update_if_necessary('do3opt_random', dist, hsh, order)   

        print 'Random improvements:', dist00, dist0, dist 
        if dist >= dist00:
            break
          
    
    for out_cnt in xrange(OUTER_N):
        
        dist00 = dist
        
        for cnt in xrange(N):
            dist0 = dist
            dist, order, num_crossed = remove_crossed_edges(N, locations, distances, closest, order, dist, MAX_EDGES)
            print '!!!! %.1f => %.1f, delta=%f, numcrossed=%d' % (dist0, dist, dist - dist0, num_crossed)  
            if dist > dist0 - 1:
                break
               
            dist = trip2(distances, order)
            hsh = np.dot(hash_base, order)
            update_if_necessary('local_search: remove_crossed_edges: cnt=%d' % cnt, dist, hsh, order)
        if num_crossed != 0:
            print '!!!!!!!!!!! num_crossed:',  num_crossed
        #assert num_crossed == 0, num_crossed
        
        if True:
            for cnt in xrange(LONG_EDGE_N):
                hsh = np.dot(hash_base, order)
                #if hsh in visited:   # Done this local search?
                #    print '*** visited' 
                #    continue
                visited.add(hsh)  
                dist0 = dist
                dist, order = local_search(N, distances, closest, dist, order, True, do3)
                dist = trip2(distances, order)
                       
                update_if_necessary('local_search: long_edges: cnt=%d' % cnt, dist, hsh, order)
                print 'Long edges: cnt=%d of %d, dist0=%f,dist=%f, diff=%f' % (cnt, LONG_EDGE_N, dist0, dist, dist-dist0)
                if dist > dist0 - 1.0:
                    print '@@@ No long edge search improvements'
                    break
        
        
        dist0 = dist  
        dist, order = local_search(N, distances, closest, dist, order, False, do3)
        print 'Local: out_cnt=%d of %d, dist0=%f,dist=%f, diff=%f' % (out_cnt, OUTER_N, dist0, dist, dist-dist0)
        print 'Local: out_cnt=%d,dist00=%f,dist=%f,diff=%f' % (out_cnt, dist00, dist, dist-dist00)
        dist = trip2(distances, order)
        hsh = np.dot(hash_base, order)
        visited.add(hsh)  
        update_if_necessary('local_search: minimum', dist, hsh, order)
        if dist > dist00 - 1.0:
            if do3:
                break
            else:
                do3 = True
    
    actual_dist = trip2(distances, order)

    assert dist > 0
    assert CLOSE(dist, actual_dist), DIFF(dist, actual_dist)
    assert len(set(order)) == len(order), start
    outer_solutions.append((dist, hsh, order))
            
    print 'Done greedy', len(outer_solutions), len(optimum_solutions)
    
    #while True:
    dist, hsh, order = optimum_solutions[-1]
    
    #hsh = np.dot(hash_base, order)   
    #if hsh in visited:   # Done this local search?
    #    continue
    dist, order = neighbor_search(N, distances, closest, visited, hash_base, dist, order, update_if_necessary)
    assert dist > 0
    dist = trip2(distances, order)
    hsh = np.dot(hash_base, order)
    visited.add(hsh) 
    update_if_necessary('neighbor_search done',  dist, hsh, order)
    
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
    dist, order = solve(path, points) 
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
path_list.reverse()

for path in path_list: # !@#$%
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

