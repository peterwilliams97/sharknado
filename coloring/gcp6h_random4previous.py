#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    Find Chromatic number and colorings of specified graphs
    
    
    TODO: 
         Find local minima faster!
          Remove color class(es) and run DSATUR during search
          Run Kempe during search
          Try actual DSATUR
          Tabu
          GA  
            Select one of best neighbours with equal probability 1/(num best neighbours)
            This will require adding values equal to current value
          
          Remove one color k, 
          Randomly reassign all vertices that were k with color k1 != k until a feasible solution
          is found
          Asssign all length 1 classes and shortest class to all other colors
          
          
          Implement a cache
          
          Implement timeouts
          
          Search by distance(X, best_X)
          
          Global solutions list
          
          set with maxlen
            set 
            dict with lru count
            
            add(elt):
                lru += 1
                dict[elt] = lru
                if elt not in set:
                    set.add(elt)
                    if len(set) >= maxlen:
                        victim = min(dict.keys(), key=lambda k: dict[k])
                        del set[victim]
                        del dict[victim]
          
"""
from __future__ import division
from itertools import count
from collections import defaultdict
import pprint
import re, os, time, shutil
from previous_best import previousXlist


VERSION = 41
print 'VERSION=%d' % VERSION

_pp = pprint.PrettyPrinter(indent=4)

def pp(s):
    _pp.pprint(s)

def xx(lst):
    return [sorted(x) for x in lst]
    
RE_PATH = re.compile('gc_(\d+)_(\d+)')
RESULTS_DIR = 'results'
 
def make_log_name(path):    
    m = RE_PATH.search(path)
    assert m, path
    try:
        os.mkdir(RESULTS_DIR)
    except:
        pass
    name = 'gc_%d_%d.%02d' % (int(m.group(1)), int(m.group(2)), VERSION)
    log_name = os.path.join(RESULTS_DIR, name)
    print 'log_name:', log_name
    return log_name
    
def get_previous(path):
    m = RE_PATH.search(path)
    if not m:
        return []
    name = m.group(0)    
    return previousXlist(name)
    

def maximal_cliques(G, n0):
    """Return maximal clique in graph G containing node n0"""
    # Q = a stack
    # V = visited nodes
    C = [set([n0])]
    C_maximal = []
    
    while C:
        c = C.pop()
        n_cliques = len(C)
        for n1 in c:
            for n2 in G[n1]:
                if n2 not in c and all(n3 in G[n2] for n3 in c):
                    C.append(c | set([n2]))
        if len(C) == n_cliques:
            C_maximal.append(c)
    
    print '@@3'
    return [frozenset(c) for c in (C_maximal + C)]       
                    
    
def maximal_clique(G, n0):   
    Q, V = [n0], set([n0])
    #print sorted(G[n0])
    
    while Q:
        #print '++', Q, list(V),
        n = Q.pop()
        #print n
        for n1 in G[n]:
            # Already visited?
            if n1 not in V and all(n2 in G[n1] for n2 in V):
                V.add(n1)
                Q.append(n1)
           # print '--', n1, Q, list(V)
    #print '@@3'
    return frozenset(V)


def all_cliques(G):
    """Return all maximal cliques in G"""
    # N^2 
    C = []
    for n in G:
        #C.extend(maximal_cliques(G, n))
        C.append(maximal_clique(G, n))
    #C = [clique(G, n) for n in G]
    
    assert len(C) == len(G)
    assert all(any(n in c for c in C) for n in G)
    
    if False:
        print '**        maximal cliques' 
        for i, c in enumerate(C):
            print '%2d : %s' % (i, sorted(c))
    #C_unique = sorted(set(C), key=lambda c: C.index(c))
    C_unique = sorted(set(C), key=lambda c: (-len(c), C.index(c)))
    assert all(any(n in c for c in C_unique) for n in G)
    
    if False:
        print '!! unique maximal cliques'
        for c in (C_unique):
            print '  %s' % sorted(c)
    
    c_all = set(C_unique[0])    
    for c in C_unique[1:]: 
        c_all.update(c)
    assert len(c_all) == len(G)    
    #for n in G:
    #    clique_n = [c for c in C_unique if n in c]
    #    assert len(clique_n) >= 2, n
        
    #assert all(any(n in G[n] in c 
    print '=' * 80
    
    if False:
        C = [C_unique[0]]
        c_all = set(C_unique[0])
        print xx(C), c_all
        for i in range(1, len(C_unique)):
            c = max(C_unique[i:], key=lambda c: -len(c & c_all))
            ll = len(c & c_all)
            C.append(c)
            c_all.update(c)
            #print i, xx(C), c, c_all, ll
    else:
        C = C_unique
    
    #assert len(C) == len(G)
    assert all(any(n in c for c in C) for n in G)
    print '-' * 80
    return C
 
 
def count_collisions(G, X):
    n_collisions = 0
    for n in G:
        for n1 in G[n]:
            if X[n1] == X[n]:
                n_collisions += 1
    return n_collisions
    
    
def normalize(X):
    """Return X as tuple with color indexes decreasing with count.
        i.e. color 0 is most plentiful
    """
    colors = sorted(set(X), key=lambda c: (-X.count(c), c))
    
    #print 'normalize', len(X), len(set(X))
    color_index = [-1] * (max(X) + 1)
    for i in set(X):
        color_index[i] = colors.index(i)
    #print colors, sorted(set(X)), color_index
    X2 = tuple([color_index[c] for c in X])  
    assert len(set(X2)) == len(set(X))
    return X2
    
def validate(G, X):    
    for n in G:
        assert X[n] >= 0, '%d %s' % (n, G[n])
        assert all(X[n] != X[n1] for n1 in G[n]), '\n%d %s\n%d %s' % (
            n, G[n], X[n], 
            [X[n1] for n1 in G[n]] ) 
    return len(set(X))        

    
def union(list_of_sets):
    unn = set([])
    for s in list_of_sets:
        unn = unn | s
    return unn

 
def populate1(G, X, Cso): 
    # Color the graph somewhat like http://carajcy.blogspot.com.au/2013/01/dsatur.html
    # Start with largest clique
    # Do a real DSTATUR !@#$
    cs, co = Cso[0]
       
    # Color the other cliques
    for _, co in Cso:
        for n in co:
            if X[n] != -1:
                continue
            neighbor_colors = set(X[i] for i in G[n])    
            for color in count():    
                if color not in neighbor_colors:
                    X[n] = color
                    break
    return tuple(X)                
    

def populate2(G, X, Cso): 
    # Color the graph like http://carajcy.blogspot.com.au/2013/01/dsatur.html
    # Do a real DSTATUR !@#$
       
    nodes = G.keys()
    nodes.sort(key=lambda n: -len(G[n]))
  
    for n in nodes:
        if X[n] != -1:
            continue
        neighbor_colors = set(X[i] for i in G[n])    
        for color in count():    
            if color not in neighbor_colors:
                X[n] = color
                break
    return tuple(X)     

flipflop = False
def populate(G, X, Cso):
    global flipflop
    X1 = populate1(G, X, Cso)
    X2 = populate2(G, X, Cso)
    c1 = len(set(X1))
    c2 = len(set(X2))
    if c1 == c2:
        flipflop = not flipflop
        return X1 if flipflop else X2
    return X1 if c1 < c2 else X2

def populate_ff(G, X, Cso):
    global flipflop
    X1 = populate1(G, X, Cso)
    X2 = populate2(G, X, Cso)
    flipflop = not flipflop
    return X1 if flipflop else X2
    
def get_Cso(G):    
    order = range(len(G))
    order.sort(key=lambda n: -len(G[n]))
    # Cliques as sets
    
    print '@@1'
    Cs = all_cliques(G)
    print '@@2'
    #assert len(Cs) == len(G)
    assert len(union(Cs)) == len(G)
    
    # Cliques as ordered lists
    Co = [sorted(c, key=lambda n: -len(G[n])) for c in Cs]
    print '>>', Co
    Cso = zip(Cs, Co)
    return Cso
    
 
def find_min_colors(G, Cso):
   
    X = [-1] * len(G)
    X = populate(G, X, Cso)    
                    
    n_colors = len(set(X))
    print n_min, n_colors, n_max
    validate(G, X)
    
    for c in sorted(set(X)):
        print '%d: %d' % (c, X.count(c))
    
    # Sort colors by most common first
    #color_order = sorted(set(X), key=lambda c: (-X.count(c), c))
    #color_order = [color_order.index(i) for i in range(n_colors)]
    #X = [color_order[c] for c in X]  
    #print color_order
    X = normalize(X)
    
    for c in sorted(set(X)):
        print '%d: %d' % (c, X.count(c))
        
    print '-' * 40
        
    return X, n_min                
 
def do_kempe(G, X0):
    X = X0[:]
    
    n_colors = len(set(X0))
    color_classes = [set([n for n, x in enumerate(X) if x == c]) for c in range(n_colors)]
    assert sum(len(cls) for cls in color_classes) == len(X)
    
    for n in G:
        for n1 in G[n]:
            assert n in G[n1], '%d %d' % (n, n1)
    
    def objective():
        return sum(len(cls)**2 for cls in color_classes)
        
    def makeX():  
        X1 = [-1] * len(X)
        for c, cls in enumerate(color_classes):
            for x in cls:
                X1[x] = c
        return X1  

    def neighbor_colors(n):
        return set(X[n1] for n1 in G[n])
    
    print '$' * 40, 'kempe'
    #print [len(cls) for cls in color_classes], objective()
     
    
    '''
        Smallest to largest color class
            #for i1 in range(n_colors -1, -1, -1):
            #    for i2 in range(i1 - 1, -1, -1):
        [22, 21, 18, 17, 15, 7] 1812
        [22, 21, 18, 14, 18, 7] 1818 (4, 3) (514, 520) True
        [22, 21, 19, 14, 17, 7] 1820 (4, 2) (648, 650) True
        [16, 21, 19, 14, 23, 7] 1832 (4, 0) (773, 785) True
        [19, 21, 19, 11, 23, 7] 1862 (3, 0) (452, 482) True
        [18, 21, 20, 11, 23, 7] 1864 (2, 0) (722, 724) True
        
        Largest to smallest color class
            #for i2 in range(1, n_colors -1):
            #    for i1 in range(0, i2):
        [22, 21, 18, 17, 15, 7] 1812
        [20, 23, 18, 17, 15, 7] 1816 (0, 1) (925, 929) True
        [20, 17, 24, 17, 15, 7] 1828 (1, 2) (853, 865) True
        [20, 18, 24, 16, 15, 7] 1830 (1, 3) (578, 580) True
        [20, 14, 24, 16, 19, 7] 1838 (1, 4) (549, 557) True
    '''
    for i2 in range(1, n_colors -1):
        for i1 in range(0, i2):
            links1 = set([n for n in color_classes[i1] if i2 in neighbor_colors(n)])
            links2 = set([n for n in color_classes[i2] if i1 in neighbor_colors(n)])
            cc1 = (color_classes[i1] - links1) | links2
            cc2 = (color_classes[i2] - links2) | links1
            before = len(color_classes[i1])**2 + len(color_classes[i2])**2
            after = len(cc1)**2 + len(cc2)**2
            if after > before:
                if len(cc2) > len(cc1):
                    color_classes[i1] = cc2
                    color_classes[i2] = cc1
                else:    
                    color_classes[i1] = cc1
                    color_classes[i2] = cc2
                print objective(), (i1, i2), (before, after), after > before # , [len(cls) for cls in color_classes]
                X = makeX()
                validate(G, X)    

    #print '$' * 40
    #X = makeX()
    
    return normalize(X)
 

def color_counts(X, n_colors):
    """Return list of numbers of elements in each color class"""
    
    # List may have zero counts
    counts = [0] * n_colors
    for c in X:
        assert 0 <= c < n_colors, c
        counts[c] += 1 
    return counts

    
def broken_counts(G, n_colors, X):
    broken = [0] * n_colors
    for n, neighbors in G.items():
        x = X[n]
        for n1 in neighbors:
            if X[n1] == x:
                broken[x] += 1
    return broken


def check_counts(G, n_colors, X, n_cc, n_bc):

    # !@#$
    return
    
    counts = color_counts(X, n_colors)
    broken = broken_counts(G, n_colors, X)
    assert counts == n_cc, '''
        %d %d
        X=%s
        counts=%s
        n_cc  =%s
    ''' % (len(G), n_colors, X, counts, n_cc)    
    assert broken == n_bc     
    
 
def perturb_by_class(G, X, Cso, n_colors):  
    #color_classes = [set([n for n, x in enumerate(X) if x == c]) for c in range(n_colors)]
    perturbations = set([])
    for c in range(n_colors):
        X1 = list(X)
        for n, x in enumerate(X):
            if x == c:
                X1[n] = -1
        X2 = populate(G, X1, Cso)
        #assert all(x >= 0 for x in X2)
        perturbations.add(normalize(X))
     
    # !@#$ Replace least common colors      
    # counts = color_counts(X, n_colors)
    # min_c = min(enumerate(counts), key=lambda x: (x[1]> 0, x[1]))
    # for c in range(n_colors):
    #    if c == min_c:
    #        continue
    #    X1 = list(X)    
           
            
    return perturbations  

    
def bc_objective(n_colors, n_cc, n_bc):
    return sum((2 * n_cc[c] * n_bc[c] - n_cc[c] ** 2) for c in range(n_colors))    
    
    
def get_score(X):    
    n_colors = len(set(X))
    counts = [0] * n_colors
    for c in X:
        assert 0 <= c < n_colors, c
        counts[c] += 1 
    return -sum(c**2 for c in counts)
    
def add_solution(solutions, G, X, count):
    """ 
        solution = (n_colors, score, count hash(nX), nX)
    """
    validate(G, X)
    n_colors = len(set(X))

    nX = normalize(X) 
    solutions.insert((n_colors, get_score(X), -count, hash(nX), nX))
    # print '$$$ best solutions', [v for v,_,_ in solutions]
    
    
  
from utils import SortedDeque
#from numba import autojit, jit, double

#@autojit
def op1(G, X, n, c, n_cc, n_bc):
    """Change color of G[n] to c and compute changes"""
    c0 = X[n]
    #print 'op1', n, c0, c, n_cc, n_bc
    #assert 0 <= c < len(n_cc), c
    #assert 0 <= c < len(n_bc), c
    #assert 0 <= c0 < len(n_cc), c0
    #assert 0 <= c0 < len(n_bc), c0 
    n_cc_c0, n_cc_c, n_bc_c0, n_bc_c = n_cc[c0], n_cc[c], n_bc[c0], n_bc[c]
    before = 2 * n_cc_c0 * n_bc_c0 - n_cc_c0**2 + 2 * n_cc_c * n_bc_c - n_cc_c**2
    n_cc_c0 -= 1
    n_cc_c += 1
    n_bc_c0 -= sum(int(X[n1] == c0) for n1 in G[n])
    n_bc_c  += sum(int(X[n1] == c)  for n1 in G[n])
    after = 2 * n_cc_c0 * n_bc_c0 - n_cc_c0**2 + 2 * n_cc_c * n_bc_c - n_cc_c**2
    return after - before, n_bc_c0, n_bc_c
    
def apply1(v, X, n_cc, n_bc, diff, n, c, n_bc_c0, n_bc_c):
    c0 = X[n]
    X1 = list(X)
    n_cc1 = n_cc[:]
    n_bc1 = n_bc[:]
    X1[n] = c
    n_cc1[c0] -= 1
    n_cc1[c] += 1
    n_bc[c0] -= n_bc_c0
    n_bc[c] -= n_bc_c
    return v + diff, X1, n_cc1, n_bc1
    
def do_search(G, visited, X, verbose=False):
    """Local search around X using sum(2*|B[i]||C[i]| - |C[i]|^2) objective
    """
    #verbose = True
    
    n_colors = len(set(X))
    # color_classes[c] = nodes with color c
    color_classes = [set([n for n, x in enumerate(X) if x == c]) for c in range(n_colors)]
    assert sum(len(cls) for cls in color_classes) == len(X)
    
    # broken_classes[c] = edges where both colors are c
    broken_classes = [set([]) for c in range(n_colors)] 
    
    # We only need counts for our objective functions
    n_cc = [len(color_classes[c]) for c in range(n_colors)]
    n_bc = [len(broken_classes[c]) for c in range(n_colors)]
       
    v = bc_objective(n_colors, n_cc, n_bc) 
    
    stack = [[(v, X, n_cc, n_bc)]]
    best_cX_list = SortedDeque([], 1000)
    
    def add_best(X):
        c = len(set(X))
        previous_best_c = best_cX_list[0][0] if best_cX_list else None 
        best_cX_list.insert((c, X))
        visited.add(hash(normalize(X)))
        if verbose and not previous_best_c or c < previous_best_c:
            print 'best_X: c=%d,X=%s' % (c, X)

    assert hash(normalize(X)) not in visited
    counter = count()
    cnt = -1
    #while stack and (cnt < 100 or len(best_cX_list) < 10):
    while stack and (cnt < 100 or not len(best_cX_list)):
        if verbose:   print '**stack %4d:' % cnt, len(stack), [len(x) for x in stack]    
        cnt = next(counter)
        L = stack.pop()
        if not L:
            if verbose: print '*done with', len(stack) + 1
            continue
         
        v, X, n_cc, n_bc = L.pop()     # L is sorted worst to best
        
        nX = normalize(X)
        hX = hash(nX)
        if hX in visited:  # Previously found a local minimum here?
            if verbose: print '        tested'
            stack.append(L)
            continue
        visited.add(hX)    
           
        # Changing to a more numerous color is usually better
        colors = sorted([i for i, c in enumerate(n_cc) if c], key=lambda x: -x)
        order = list(G.keys())
        #print '*', len(order), len(colors), n_colors
        assert len(colors) == len(set(X))
        # Changing from a lass color is usually better
        order.sort(key=lambda n: n_cc[X[n]])
        moves_1 = []
        number_tested = 0
        for n in order:
            for c in colors:
                if c == X[n]: continue
                       
                diff, n_bc_c0, n_bc_c = op1(G, X, n, c, n_cc, n_bc)
                if diff >= 0: # !@#$
                    X1 = list(X)
                    X1[n] = c
                    #visited.add(hash(normalize(X1)))
                    number_tested += 1
                    visited.add(hash(tuple(X1)))
                    continue
                moves_1.append((n, c, diff, n_bc_c0, n_bc_c))
             
        if verbose:
            print '*** number_tested:', number_tested, len(visited)
            print '    --- moves_1', len(moves_1), [x[0] for x in moves_1]        
        if not moves_1:
            # Local minimum
            add_best(nX)
            if verbose:   print '    **** mininum', len(set(X)), best_cX_list[0][0]  
            
            continue
            
        biggest_diff = min(diff for _,_,diff,_,_ in moves_1)        
        moves_1 = [x for x in moves_1 if x[2] <= biggest_diff]
        if verbose:
            print '    +++ moves_1', len(moves_1), [x[2] for x in moves_1]
        assert moves_1
        L1 = [apply1(v, X, n_cc, n_bc, diff, n, c, n_bc_c0, n_bc_c) 
                    for n, c, diff, n_bc_c0, n_bc_c in moves_1]
        L1.sort()            
        stack.append(L)
        stack.append(L1)
    
    if verbose or True:
        print 'best_cX_list', cnt, [c for c,_ in best_cX_list]
   
    return [x[1] for x in best_cX_list]                    

def repopulate(G, visited, X0, Cso2, target_score, visited2, fraction):  
    #print 'repopulate ------------------------'
    #X = list(X0)
    color_counts = defaultdict(int)
    for c in X0:
        color_counts[c] += 1
    assert sum(color_counts.values()) == len(X0)    
    assert len(color_counts) == len(set(X0))
    #colors = color_counts.keys()
    n_colors = len(color_counts)
    # excluded = set([c for c, k in color_counts.items() if k <= n_colors/2])
    max_color = max(color_counts.values())
    
    excluded0 = set([])
    excluded1 = set([c for c, k in color_counts.items() if k <= 1]) 
    excluded2 = set([c for c, k in color_counts.items() if k <= 2])  
    excluded3 = set([c for c, k in color_counts.items() if k <= 3])       
    #print '** excluded1', len(excluded1), (len(color_counts), len(X)), max(color_counts.values())
    #print '** excluded2', len(excluded2),
    #print [(c,color_counts[c]) for c in sorted(color_counts, key=lambda x: -color_counts[x])]
      
    hX0 = hash(X0)
    visited.add(hX0)
    best_score = 0
    while True:
        for count in xrange(100):
            for excluded in excluded1, excluded2, excluded3:
                for ii in range(20):
                    X = list(X0)
                    for i in range(len(X)):
                        #if color_counts[X[i]] in excluded or random.randrange(1, 3) == 1:
                        if color_counts[X[i]] in excluded or random.random() <= fraction:  # !@#$
                            X[i] = -1
                    X = populate2(G, X, Cso2) if ii % 2 == 0 else populate1(G, X, Cso2)
                    X = normalize(X)
                    hX = hash(X)
                    if hX not in visited and hX not in visited2: # !@#$
                        break
                    visited.add(hX)    
                if hX not in visited:
                    break    
            score = get_score(X)
            if score < best_score:
                best_score = score
                best_X = X
                if score <= target_score:
                    break
        #print 'X0', hash(X0), X0
        #print 'X', hash(X), X
        #print '---- repopulated'
        X = best_X
        X = normalize(X)
        hX = hash(X)
        if hX not in visited:
            break
        fraction *= 1.9
        #print '$$ loosening repopulate to fraction=%f' % fraction 
        if fraction >= 1.9:
            return None
        
    assert hX not in visited
    visited.add(hX) 
    return X    

import random    

previous_X = [
(55, 15, 73, 30, 51, 73, 10, 69, 87, 94, 6, 15, 93, 11, 28, 50, 44, 3, 49, 83, 83, 52, 93, 90, 16, 17, 104, 35, 9, 28, 74, 6, 69, 61, 50, 13, 36, 84, 18, 52, 14, 3, 14, 23, 51, 41, 78, 16, 77, 34, 30, 24, 41, 29, 61, 36, 99, 1, 21, 40, 35, 52, 26, 1, 53, 63, 94, 91, 100, 3, 77, 101, 7, 38, 76, 38, 50, 75, 103, 29, 64, 66, 40, 15, 2, 16, 14, 24, 18, 39, 6, 72, 63, 94, 79, 66, 6, 84, 21, 97, 80, 93, 60, 43, 5, 31, 43, 24, 41, 39, 52, 70, 8, 19, 5, 62, 26, 80, 70, 75, 76, 37, 22, 11, 81, 42, 38, 20, 48, 42, 68, 101, 46, 26, 52, 17, 81, 61, 45, 3, 5, 64, 91, 93, 82, 22, 33, 58, 56, 9, 35, 102, 12, 50, 30, 10, 68, 61, 71, 20, 14, 10, 18, 72, 24, 9, 13, 70, 2, 7, 88, 21, 15, 83, 64, 18, 33, 13, 41, 11, 12, 44, 36, 23, 79, 29, 46, 46, 9, 45, 27, 88, 88, 94, 93, 26, 78, 98, 0, 7, 23, 60, 58, 60, 27, 99, 59, 30, 23, 22, 4, 3, 45, 62, 88, 38, 40, 56, 4, 74, 102, 19, 58, 47, 16, 26, 10, 47, 12, 87, 102, 44, 67, 27, 6, 34, 0, 8, 51, 62, 57, 80, 30, 79, 94, 57, 49, 58, 78, 32, 58, 51, 76, 27, 14, 47, 47, 15, 104, 8, 10, 38, 21, 73, 39, 23, 72, 80, 81, 72, 70, 7, 66, 6, 49, 84, 0, 100, 32, 4, 75, 90, 13, 37, 81, 4, 22, 93, 2, 91, 101, 87, 52, 86, 0, 90, 96, 25, 65, 34, 31, 7, 36, 35, 95, 11, 85, 43, 103, 0, 81, 66, 95, 11, 17, 96, 40, 36, 5, 32, 77, 2, 48, 85, 2, 3, 36, 24, 3, 5, 53, 95, 9, 92, 52, 98, 7, 78, 25, 35, 1, 11, 33, 7, 100, 36, 18, 65, 3, 44, 40, 31, 46, 100, 56, 12, 25, 87, 16, 55, 4, 50, 52, 58, 91, 33, 74, 56, 12, 57, 53, 55, 38, 70, 10, 65, 0, 13, 40, 19, 12, 40, 31, 62, 7, 17, 26, 50, 58, 10, 64, 73, 54, 72, 44, 17, 17, 22, 32, 47, 62, 24, 84, 50, 59, 10, 46, 17, 57, 24, 47, 21, 38, 37, 48, 37, 67, 95, 56, 54, 9, 66, 96, 23, 54, 0, 33, 21, 100, 20, 87, 17, 13, 2, 35, 26, 52, 66, 1, 4, 17, 72, 74, 98, 89, 18, 76, 31, 91, 7, 25, 98, 9, 0, 64, 98, 3, 42, 55, 71, 70, 8, 44, 60, 4, 87, 21, 77, 93, 28, 19, 88, 44, 69, 16, 76, 29, 68, 63, 44, 74, 28, 88, 64, 22, 31, 60, 67, 84, 37, 89, 45, 64, 98, 77, 92, 16, 69, 85, 99, 48, 13, 49, 53, 63, 7, 46, 42, 89, 10, 75, 96, 13, 84, 51, 2, 106, 98, 57, 82, 39, 69, 27, 66, 51, 95, 72, 30, 47, 76, 72, 21, 104, 25, 41, 11, 99, 67, 58, 39, 23, 63, 89, 59, 53, 55, 28, 94, 82, 85, 97, 38, 46, 54, 5, 9, 71, 8, 2, 20, 36, 25, 32, 83, 3, 82, 22, 28, 83, 41, 71, 68, 54, 46, 14, 83, 60, 59, 36, 20, 5, 63, 55, 20, 86, 1, 102, 25, 66, 42, 68, 79, 4, 2, 41, 42, 68, 40, 92, 25, 21, 44, 74, 42, 61, 1, 6, 29, 92, 51, 83, 81, 87, 73, 64, 78, 15, 60, 48, 10, 1, 39, 5, 61, 20, 76, 38, 32, 29, 28, 82, 95, 23, 24, 65, 86, 52, 82, 81, 15, 27, 62, 22, 33, 59, 80, 29, 84, 29, 68, 15, 71, 79, 24, 53, 84, 63, 44, 71, 31, 25, 67, 73, 4, 12, 90, 1, 41, 35, 29, 82, 18, 48, 24, 101, 1, 67, 92, 39, 47, 43, 21, 89, 67, 68, 21, 100, 4, 9, 0, 91, 61, 105, 75, 68, 46, 69, 33, 48, 65, 89, 25, 26, 11, 59, 33, 75, 58, 35, 49, 76, 42, 26, 14, 29, 9, 77, 57, 25, 9, 55, 55, 17, 79, 51, 18, 28, 57, 91, 30, 56, 43, 59, 90, 70, 90, 40, 34, 97, 37, 3, 78, 70, 71, 40, 56, 65, 13, 81, 24, 102, 86, 105, 11, 6, 86, 37, 74, 26, 79, 48, 22, 27, 43, 49, 34, 97, 18, 8, 57, 92, 12, 57, 38, 8, 34, 80, 53, 34, 32, 82, 54, 94, 30, 31, 1, 14, 20, 75, 0, 31, 8, 60, 54, 43, 65, 32, 49, 85, 35, 22, 86, 10, 79, 48, 80, 75, 42, 85, 16, 63, 22, 63, 27, 33, 99, 34, 17, 86, 2, 62, 31, 79, 97, 2, 96, 11, 45, 28, 92, 51, 16, 80, 30, 23, 27, 15, 39, 60, 55, 77, 51, 18, 78, 103, 61, 14, 19, 41, 50, 33, 96, 35, 0, 72, 12, 66, 19, 37, 53, 11, 101, 5, 13, 78, 105, 12, 5, 70, 90, 37, 76, 30, 49, 47, 6, 45, 8, 14, 74, 34, 88, 5, 97, 96, 69, 23, 20, 31, 62, 3, 8, 19, 71, 45, 69, 67, 75, 1, 103, 36, 46, 45, 33, 67, 32, 15, 19, 53, 78, 43, 54, 77, 56, 89, 53, 34, 4, 20, 8, 65, 69, 65, 19, 4, 54, 71, 12, 26, 23, 7, 59, 61, 87, 6, 91, 55, 19, 101, 77, 85, 13, 42, 95, 97, 20, 30, 85, 41, 39, 54, 5, 39, 73, 27, 15, 50, 90, 99, 37, 73, 2, 34, 43, 16, 16, 0, 59, 47, 18, 8, 32, 45, 7, 83, 32, 92, 88, 43, 49, 27, 74, 19, 14, 6, 48, 56, 73, 64, 50, 28, 1, 28, 29, 49, 45, 86, 6, 62, 89),

(56, 15, 74, 30, 52, 74, 10, 70, 87, 94, 6, 15, 93, 11, 28, 51, 43, 3, 50, 83, 83, 53, 93, 90, 16, 17, 104, 33, 9, 28, 75, 6, 70, 61, 51, 13, 34, 84, 18, 53, 14, 3, 14, 23, 52, 40, 79, 16, 78, 81, 30, 24, 40, 29, 61, 34, 99, 1, 21, 39, 33, 53, 26, 1, 54, 63, 94, 91, 100, 3, 78, 102, 7, 36, 77, 36, 51, 76, 103, 29, 64, 66, 39, 15, 2, 16, 14, 24, 18, 37, 6, 73, 63, 94, 80, 66, 6, 84, 21, 97, 47, 93, 60, 42, 5, 31, 42, 24, 40, 37, 53, 71, 8, 19, 5, 62, 26, 47, 71, 76, 77, 35, 22, 11, 68, 41, 36, 20, 49, 41, 69, 102, 46, 26, 53, 17, 68, 61, 45, 3, 5, 64, 91, 93, 82, 22, 44, 58, 38, 9, 33, 101, 12, 51, 30, 10, 69, 61, 72, 20, 14, 10, 18, 73, 24, 9, 13, 71, 2, 7, 88, 21, 15, 83, 64, 18, 68, 13, 40, 11, 12, 43, 34, 23, 80, 29, 46, 46, 9, 45, 27, 88, 88, 94, 93, 26, 79, 98, 0, 7, 23, 60, 58, 60, 27, 99, 59, 30, 23, 22, 4, 3, 45, 62, 88, 36, 39, 38, 4, 75, 101, 19, 58, 48, 16, 26, 10, 48, 12, 87, 101, 43, 67, 27, 6, 81, 0, 8, 52, 62, 57, 47, 30, 80, 94, 57, 50, 58, 79, 32, 58, 52, 77, 27, 14, 48, 48, 15, 104, 8, 10, 36, 21, 74, 37, 23, 73, 47, 68, 73, 71, 7, 66, 6, 50, 84, 0, 100, 32, 4, 76, 90, 13, 35, 68, 4, 22, 93, 2, 91, 102, 87, 53, 86, 0, 90, 96, 25, 65, 81, 31, 7, 34, 33, 95, 11, 85, 42, 103, 0, 68, 66, 95, 11, 17, 96, 39, 34, 5, 32, 78, 2, 49, 85, 2, 3, 34, 24, 3, 5, 54, 95, 9, 92, 53, 98, 7, 79, 25, 33, 1, 11, 44, 7, 100, 34, 18, 65, 3, 43, 39, 31, 46, 100, 38, 12, 25, 87, 16, 56, 4, 51, 53, 58, 91, 44, 75, 38, 12, 57, 54, 56, 36, 71, 10, 65, 0, 13, 39, 19, 12, 39, 31, 62, 7, 17, 26, 51, 58, 10, 64, 74, 55, 73, 43, 17, 17, 22, 32, 48, 62, 24, 84, 51, 59, 10, 46, 17, 57, 24, 48, 21, 36, 35, 49, 35, 67, 95, 38, 55, 9, 66, 96, 23, 55, 0, 44, 21, 100, 20, 87, 17, 13, 2, 33, 26, 53, 66, 1, 4, 17, 73, 75, 98, 89, 18, 77, 31, 91, 7, 25, 98, 9, 0, 64, 98, 3, 41, 56, 72, 71, 8, 43, 60, 4, 87, 21, 78, 93, 28, 19, 88, 43, 70, 16, 77, 29, 69, 63, 43, 75, 28, 88, 64, 22, 31, 60, 67, 84, 35, 89, 45, 64, 98, 78, 92, 16, 70, 85, 99, 49, 13, 50, 54, 63, 7, 46, 41, 89, 10, 76, 96, 13, 84, 52, 2, 106, 98, 57, 82, 37, 70, 27, 66, 52, 95, 73, 30, 48, 77, 73, 21, 104, 25, 40, 11, 99, 67, 58, 37, 23, 63, 89, 59, 54, 56, 28, 94, 82, 85, 97, 36, 46, 55, 5, 9, 72, 8, 2, 20, 34, 25, 32, 83, 3, 82, 22, 28, 83, 40, 72, 69, 55, 46, 14, 83, 60, 59, 34, 20, 5, 63, 56, 20, 86, 1, 101, 25, 66, 41, 69, 80, 4, 2, 40, 41, 69, 39, 92, 25, 21, 43, 75, 41, 61, 1, 6, 29, 92, 52, 83, 68, 87, 74, 64, 79, 15, 60, 49, 10, 1, 37, 5, 61, 20, 77, 36, 32, 29, 28, 82, 95, 23, 24, 65, 86, 53, 82, 68, 15, 27, 62, 22, 44, 59, 47, 29, 84, 29, 69, 15, 72, 80, 24, 54, 84, 63, 43, 72, 31, 25, 67, 74, 4, 12, 90, 1, 40, 33, 29, 82, 18, 49, 24, 102, 1, 67, 92, 37, 48, 42, 21, 89, 67, 69, 21, 100, 4, 9, 0, 91, 61, 105, 76, 69, 46, 70, 44, 49, 65, 89, 25, 26, 11, 59, 44, 76, 58, 33, 50, 77, 41, 26, 14, 29, 9, 78, 57, 25, 9, 56, 56, 17, 80, 52, 18, 28, 57, 91, 30, 38, 42, 59, 90, 71, 90, 39, 81, 97, 35, 3, 79, 71, 72, 39, 38, 65, 13, 68, 24, 101, 86, 105, 11, 6, 86, 35, 75, 26, 80, 49, 22, 27, 42, 50, 47, 97, 18, 8, 57, 92, 12, 57, 36, 8, 81, 47, 54, 81, 32, 82, 55, 94, 30, 31, 1, 14, 20, 76, 0, 31, 8, 60, 55, 42, 65, 32, 50, 85, 33, 22, 86, 10, 80, 49, 47, 76, 41, 85, 16, 63, 22, 63, 27, 44, 99, 81, 17, 86, 2, 62, 31, 80, 97, 2, 96, 11, 45, 28, 92, 52, 16, 47, 30, 23, 27, 15, 37, 60, 56, 78, 52, 18, 79, 103, 61, 14, 19, 40, 51, 44, 96, 33, 0, 73, 12, 66, 19, 35, 54, 11, 44, 5, 13, 79, 105, 12, 5, 71, 90, 35, 77, 30, 50, 48, 6, 45, 8, 14, 75, 81, 88, 5, 97, 96, 70, 23, 20, 31, 62, 3, 8, 19, 72, 45, 70, 67, 76, 1, 103, 34, 46, 45, 47, 67, 32, 15, 19, 54, 79, 42, 55, 78, 38, 89, 54, 38, 4, 20, 8, 65, 70, 65, 19, 4, 55, 72, 12, 26, 23, 7, 59, 61, 87, 6, 91, 56, 19, 102, 78, 85, 13, 41, 95, 97, 20, 30, 85, 40, 37, 55, 5, 37, 74, 27, 15, 51, 90, 99, 35, 74, 2, 81, 42, 16, 16, 0, 59, 48, 18, 8, 32, 45, 7, 83, 32, 92, 88, 42, 50, 27, 75, 19, 14, 6, 49, 38, 74, 64, 51, 28, 1, 28, 29, 50, 45, 86, 6, 62, 89)

]
 
def solve_(n_nodes, n_edges, edges, previous_solutions,
            visited_starting, visited_minimum, visited_tested):
    """
        Return chromatic number for graph
        n_nodes:number of nodes in graph
        n_edges: number of edges in graph
        edges: list of all edges 
        Returns: 
            number of colors, list of node colors in same order as nodes, optimal?
    """
    # G = node: edges from node
    G = defaultdict(list)
    for n1, n2 in edges:
        G[n1].append(n2)
        G[n2].append(n1)
        
    for n in G:
        G[n] = frozenset(G[n])
        
    Cso = get_Cso(G)    
        
    # Lower and upper bounds on number of colors    
    n_min = len(Cso[0][0])
    n_max = len(G)
    print 'n_min=%d,n_max=%d' % (n_min, n_max)
    
    MAX_SOLUTIONS =  1000
    MAX_VISITED = 50 * 1000 * 1000 
     
    assert MAX_VISITED >= 3 * MAX_SOLUTIONS
    
    #candidate_solutions = SortedDeque([], MAX_SOLUTIONS)
    optimized_solutions = SortedDeque([], MAX_SOLUTIONS)
    #visited_starting, visited_minimum, visited_tested = set([]), set([]), set([])
            
    X = [-1] * len(G)
    X = populate_ff(G, X, Cso) 

    X = normalize(X)
    hX = hash(X)

    visited_minimum.discard(hX) 
    visited_tested.discard(hX) 
    visited_starting.discard(hX) 
            
    def print_best():
           
        #print '=' * 80
        #print log_name
        if os.path.exists(log_name):
            shutil.copy(log_name, log_name + '.old')
        with open(log_name , 'wt') as f:
            f.write('VERSION=%d\n' % VERSION)
            f.write('log_name=%s\n' % log_name)
            
            f.write('fraction_changed=%f\n' % fraction_changed)
            
            for name, solutions in ('optimized_solutions', optimized_solutions), ('previous_solutions', previous_solutions) :
                f.write('%s\n' % ('-' * 80))
                f.write('%s=%d\n' % (name, len(solutions)))
                if not solutions:
                    continue
                max_min = 0
                min_k = solutions[0][0]
                for i, soln in enumerate(solutions):
                    if soln[0] > min_k:
                        break 
                    max_min = i 

                f.write('n_min=%d,n_max=%d\n' % (n_min, n_max))
                f.write('count=%d,visited_starting=%d,visited_tested=%d,visited_minimum=%d\n' % (count, 
                        len(visited_starting), len(visited_tested), len(visited_minimum)))
                f.write('lowest number colors: %d : %s\n' % (len(solutions), 
                    [solutions[i][0:3] for i in range(min(len(solutions), 50))]))
                f.write('best solutions: %d of %d\n' % (max_min + 1, len(solutions)))
                for i in range(min(100, max_min + 1)):
                    X = solutions[i][-1]
                    optimal = len(set(X)) == n_min
                    f.write('%3d: %s: %s\n' % (i, optimal, solutions[i]))
                
    n_colors = len(G)
    last_report_time = time.time()
    #while len(solutions) < MAX_SOLUTIONS:
    
    if True:
        for i, X in enumerate(previous_X):
            hX = hash(normalize(X))
            if hX not in visited_minimum:
                add_solution(optimized_solutions, G, X, -1000 - i) 
                visited_minimum.add(hX)
            print 'previous_X', len(previous_X), len(optimized_solutions), len(X), len(set(X))       
     
    print '^^^', len(optimized_solutions), len(visited_minimum)
    
    fraction_changed = 0.33
    local_minimum_count = 0
    
    for count in xrange(10**9):   
        
        #if count % 100 == 0:
        #    fraction_changed = 0.33
        
        fraction_changed *= 0.9
        if fraction_changed < 0.05:
            fraction_changed = 0.05
       
        if len(visited_minimum) >= MAX_VISITED: # 1000 * 1000:
            v1 = len(visited_minimum)
            visited_minimum.clear()
            v2 = len(visited_minimum)
            visited_minimum = visited_minimum | set(s[-2] for s in optimized_solutions)
            print 'resetting visited', v1, v2, len(visited_minimum)
            
        #if len(solutions) % 1000 == 2:
        if time.time() > last_report_time + 60:
            print_best()
            last_report_time = time.time()
        
        #X = do_kempe(G, X)
        
        # Replenish candidate_solutions
        target_score = optimized_solutions[0][0] if optimized_solutions else 0
        candidate_index = 0
        
        # Caught in a local minimum so randomize
        is_local_minimum = False
        LOCAL_MINIMUM_THRESHOLD = 10
        if len(optimized_solutions) > LOCAL_MINIMUM_THRESHOLD:
            n_lowest, score_lowest = optimized_solutions[0][:2]
            if len(optimized_solutions) > n_lowest:
                if optimized_solutions[n_lowest][1] == optimized_solutions[0][1]: 
                    print 'local minimum', (n_lowest, score_lowest), local_minimum_count
                    is_local_minimum = True
                      
        if is_local_minimum:
            fraction_changed = 0.5
            local_minimum_count += 1
            if local_minimum_count > max(10, n_colors * 2):
                print 'Too long on local minimum'    
                break
        else:
            local_minimum_count = 0
          
         
        #for ii in range(1000):
        foundX = True
        while candidate_index < min(len(optimized_solutions), n_colors): 
            foundX = False
            X = repopulate(G, visited_starting, X, Cso, target_score, visited_minimum, fraction_changed)  
            if X is None:
                #print '!! Repopulate failed !!!', candidate_index, len(optimized_solutions)
                candidate_index += 1  
                target_score = optimized_solutions[candidate_index][0]
                X = optimized_solutions[candidate_index][-1]
                continue
            X = normalize(X)    
            hX = hash(X)
            if hX not in visited_minimum and hX not in visited_tested:
                foundX = True
                break
            print 'repopulating'  
        if not foundX:
            print '%% %% No decent neighbours', candidate_index, (len(optimized_solutions), n_colors)
            break
            
        #print X, ':', fraction_changed, ':', len(optimized_solutions)
        print fraction_changed, ':', (candidate_index, len(optimized_solutions)), n_colors, '::', 
            
        optimal = len(set(set(X))) == n_min
        if not optimal:
            X_list = do_search(G, visited_minimum, X)
            for X in X_list:
                n_col = len(set(X))
                optimal = n_col == n_min
                #print 'i=%d,optimal=%s' % (len(solutions), optimal)
                add_solution(optimized_solutions, G, X, count)
                add_solution(previous_solutions, G, X, count)
                if n_col < n_colors:
                    print 'n_colors %d => %d, count=%d, visited=%s, solutions=%s, fraction_changed=%f' % (n_colors, 
                        n_col, count, 
                        (len(visited_starting), len(visited_tested), len(visited_minimum)),
                        (len(optimized_solutions)),
                        fraction_changed)
                    n_colors = n_col
                if optimal:
                    break    
        if optimal:
            break
            
        if len(optimized_solutions) >= 1:
            if random.randrange(0, 3) != 0:
                X = optimized_solutions[0][-1]
            else:    
                max_all = len(optimized_solutions) - 1
                max_min = 0
                min_k = optimized_solutions[0][0]
                for i, soln in enumerate(optimized_solutions):
                    if soln[0] > min_k:
                        break
                    max_min = i
                if random.randrange(0, 3) != 0:
                    max_i = max_min
                else:
                    max_i = max_all
                X = optimized_solutions[random.randrange(0, max_i+1)][-1]    
            
       
        #visited = [hash(s[1]) for s in solutions]
       
        if False:
            print '------------'
            print 'solutions', len(solutions), [(solutions[i][0], 
                        (solutions[i][2], len(solutions)-solutions[i][2] -1), hash(s[1])) 
                            for i in range(min(len(solutions), 40))]
            print 'visited', len(visited), visited   
        
    
    print '*^*Done'
    print_best()
    X = optimized_solutions[0][-1]
    return n_colors, X, optimal
   
    

def solve(n_nodes, n_edges, edges):
    n_colors_best = n_edges
    previous_solutions = SortedDeque([], 1000)
    
    while True:
        visited_starting, visited_minimum, visited_tested = set([]), set([]), set([])
        n_colors, X, optimal = solve_(n_nodes, n_edges, edges, previous_solutions,
                    visited_starting, visited_minimum, visited_tested)
        print '=' * 80
        print n_colors,  optimal
        if optimal:
            return n_colors, X, optimal 
        print '=' * 80
        print '*** Restarting'
        
        
    
def solveIt(inputData):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = inputData.split('\n')

    firstLine = lines[0].split()
    nodeCount = int(firstLine[0])
    edgeCount = int(firstLine[1])

    edges = []
    for i in range(1, edgeCount + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))

    # build a trivial solution
    # every node has its own color
    solution = range(0, nodeCount)

    n_colors, colors, optimal = solve(nodeCount, edgeCount, edges)
    
    # prepare the solution in the specified output format
    outputData = '%d %d\n' % (n_colors, optimal)
    outputData += ' '.join(map(str, colors))

    return outputData


import sys

import glob
mask = sys.argv[1]
results_f = open('results.all', 'wt')
path_list = sorted(glob.glob(mask))

for path in path_list:
    #print '-' * 80
    print path 
    log_name = make_log_name(path)
    #previous_X = get_previous(path)
 
    results_f.write('%s' % path)
    results_f.flush()
    with open(path, 'rt') as f:
        data = f.read()
    outputData = solveIt(data)
    print outputData
    results_f.write(' %s\n' % outputData)
    results_f.flush()
    print '*' * 80
 
exit()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        process_file(sys.argv[1])
    else: 
        print 'test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)'




if __name__ == '__main__':
    if len(sys.argv) > 1:
        fileLocation = sys.argv[1].strip()
        inputDataFile = open(fileLocation, 'r')
        inputData = ''.join(inputDataFile.readlines())
        inputDataFile.close()
        print solveIt(inputData)
    else:
        print 'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)'

