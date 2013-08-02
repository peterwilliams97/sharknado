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

# 40 forwards, 41 backwards
VERSION = 40

MAX_CLOSEST = 20
MAX_N =  30 * 1000
DEBUG = False
EPSILON = 1e-6
RANDOM_SEED = 120 # Not the Nelson!

print 'VERSION=%d' % VERSION
print 'MAX_CLOSEST=%d' % MAX_CLOSEST
print 'MAX_N=%d' % MAX_N

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
        for i in xrange(1, N):
            for j in xrange(i):
                assert distances[i, j] > 0.0
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
    

    print '@@4'    
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
    saved_solutions[path] = (dist, order) 
    history = HISTORY % VERSION
    print 'Writing history:', history, sys.argv[0]
    with open(history, 'wt') as f:
        f.write('prog="%s"\n' % sys.argv[0])
        f.write('VERSION=%d\n' % VERSION)
        f.write('MAX_CLOSEST=%d\n' % MAX_CLOSEST)
        f.write('MAX_N=%d\n' % MAX_N)
        f.write('RANDOM_SEED=%d\n' % RANDOM_SEED)
        f.write('DEBUG=%s\n' % DEBUG)
        f.write('EPSILON=%s\n' % EPSILON)
        f.write('saved_paths = %s\n' % repr(sorted(saved_paths)))
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
def find_2_3opt_min(N, distances, closest, order):
    
    N1 = N - 1
    N2 = N - 2
    #N4 = N - 4
    M = min(N1, MAX_CLOSEST)
    M2 = int(min(N1, math.sqrt(MAX_CLOSEST)))
        
    delta_ = 0.0
    p1_, p2_, p3_ = -1, -1, -1
    opt3_i = -1
    opt3deltas = np.zeros(4)
        
    counter = count()
    
    for p1 in xrange(N - 4):
        # for p2 in xrange(p1+2, N - 2):
        #n2 = 0
        n2cnt = 0
        closest1 = closest[p1]
        for n2 in xrange(N2):
            cnt = next(counter)
            if cnt % 1000000 == 100000:
                print 'cnt2=%d,(p1=%d,p2=%d),n2=%d,delta_=%.1f,dist=%.1f' % (cnt, p1, p2, n2, delta_, dist+delta)
           
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
            if n2cnt > M: break 
       
    counter2 = count()
    
    done_p3 = set()     
    for p1 in xrange(N - 6):
        #for p2 in xrange(p1+2, N - 4):
        #n2 = 0
        #for p2 in closest[p1]:
        #    if p2 < p1 + 2: continue
        #    if n2 >= M: break
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
                    print 'cnt3=%d,(p1=%d,p2=%d,p3=%d),n2=%d,n3=%d,delta_=%.1f,dist=%.1f' % (cnt, p1, p2, p3, n2, n3, delta_, dist+delta)
                    
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
                    opt3deltas[3]= distances[w1, w4] + distances[w2, w5] + distances[w3, w6] - bf 
                   
                    for i in xrange(4):
                        if opt3deltas[i] < delta_:
                            delta_ = opt3deltas[i]
                            opt3_i = i
                            p1_, p2_, p3_ = p1, p2, p3
                    
                    n3cnt += 1
                if n3cnt > M2: 
                    #print (n3cnt,),
                    break 
                
            n2cnt += 1
            
            if n2cnt > M2: break
        #print ('*', n2cnt, n2, cnt)
            
    return delta_, p1_, p2_, p3_, opt3_i                              
    #return delta_, np.array([p1_, p2_, p3_, opt3_i])                   

   
def do3opt_local(N, distances, closest, dist, order):
    
    #assert len(set(order)) == len(order)
    delta, p1, p2, p3, opt3_i = find_2_3opt_min(N, distances, closest, order, dist)
    
    dist1, order1 = dist, order
    #print 'best:', best 

    if delta < 0.0:
        if p3 < 0: # 2-opt
            dist1, order1 = do2opt(N, distances, dist, order, delta, p1, p2, p1+1, p2+1)
        else: # 3-opt    
            dist1 = dist + delta
            order1 = do3_all[opt3_i](N, order, ((p1, p2, p3), (p1+1, p2+1, p3+1)))
        
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
        
def solve(path, points):
    """Return traversal order of points that minimizes distance travelled"""
    
    
    N, locations, distances, closest = precalculate(points)
        
    hash_base = np.random.randint(10**4, 10**6, N)
    visited = set()
    
    outer_solutions = []
    optimum_solutions = []
    
    
    NUM_GREEDY = (MAX_N + N - 1)//N
    NUM_GREEDY = min(NUM_GREEDY, N)
    print 'NUM_GREEDY=%d' % NUM_GREEDY 
    
    last_save_time = [time.time()]
    
    def update_if_necessary(title):
        # Save our valuable result before we assert
        if not optimum_solutions or dist < optimum_solutions[-1][0]:
            actual_dist = trip2(distances, order)
            optimum_solutions.append((actual_dist, hsh, order))
            print 'best %s:' % title, optimum_solutions[-1][0]
            tm = time.time()
            if  tm > last_save_time[0] + 60 * 5:
                print 'saving:',  tm - last_save_time[0] 
                save_solution(path, points, actual_dist, order)
                last_save_time[0] = tm # !@#$
            if not CLOSE(dist, actual_dist):
                print '**********************', DIFF(dist, actual_dist)    
    
    start_list = range(N)
    random.shuffle(start_list)
    for istart, start in enumerate(start_list[:NUM_GREEDY]):
        print '$%d of %d: %d:' % (istart, NUM_GREEDY, start),
        dist, order = populate_greedy(N, distances, closest, start)
        print '%.2f)' % dist, 
           
        normalize(N, order)
        hsh = np.dot(hash_base, order)   
        if hsh in visited:   # Done this local search?
            continue
        visited.add(hsh) 
        
        update_if_necessary('greedy')
         
        assert len(set(order)) == len(order), start
        #actual_dist = trip2(distances, order)
        #assert CLOSE(dist, actual_dist), DIFF(dist, actual_dist)
       
        dist, order = local_search(N, distances, closest, dist, order)
        hsh = np.dot(hash_base, order)
        visited.add(hsh)  
        update_if_necessary('local_search')
        
        actual_dist = trip2(distances, order)

        assert dist > 0
        assert CLOSE(dist, actual_dist), DIFF(dist, actual_dist)
        assert len(set(order)) == len(order), start
        outer_solutions.append((dist, hsh, order))
        
            
    print 'Done greedy', len(outer_solutions), len(optimum_solutions)
    
    for dist, hsh, order in outer_solutions:
        
        hsh = np.dot(hash_base, order)   
        if hsh in visited:   # Done this local search?
            continue
        dist, order = search(N, distances, visited, hash_base, dist, order)
        assert dist > 0
        update_if_necessary('search', True)
    
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
#path_list.reverse()

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

