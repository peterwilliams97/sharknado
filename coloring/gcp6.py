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

VERSION = 5
print 'VERSION=%d' % VERSION

_pp = pprint.PrettyPrinter(indent=4)

def pp(s):
    _pp.pprint(s)

def xx(lst):
    return [sorted(x) for x in lst]
    

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
    
    
def add_solution(solutions, G, X):
    n_colors = len(set(X))
    solutions.insert((n_colors, normalize(X)))
    print '$$$ best solutions', [solutions[i][0] for i in range(min(len(solutions), 20))]
    return v    
    
DEQU_LEN = 1000 
  
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
    
def do_search(G, X, verbose=False):
    """Local search around X using sum(2*|B[i]||C[i]| - |C[i]|^2) objective
    """
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
    tested = set()
    
    best_cX_list = SortedDeque([], 1000)
    
    def add_best(X):
        c = len(set(X))
        previous_best_c = best_cX_list[0][0] if best_cX_list else None 
        best_cX_list.insert((c, X))
        if verbose and not previous_best_c or c < previous_best_c:
            print 'best_X: c=%d,X=%s' % (c, X)
            

    add_best(normalize(X))
    counter = count()
    cnt = 0
    while stack and (cnt < 100 or len(best_cX_list) < 10):
        if verbose:
            print '**stack %4d:' % cnt, len(stack), [len(x) for x in stack]    
        cnt = next(counter)
        L = stack.pop()
        if not L:
            #print '*done with', len(stack) + 1
            continue
         
        v, X, n_cc, n_bc = L.pop()     # L is sorted worst to best
        
        nX = normalize(X)
        hX = hash(nX)
        #print 'tested', hX, tested
        #if hX in tested:
        #    print '        tested'
        #    stack.append(L)
        #    continue
               
        
        # Changing to a more numerous color is usually better
        colors = sorted([i for i, c in enumerate(n_cc) if c], key=lambda x: -x)
        order = list(G.keys())
        #print '*', len(order), len(colors), n_colors
        assert len(colors) == len(set(X))
        # Changing from a lass color is usually better
        order.sort(key=lambda n: n_cc[X[n]])
        moves_1 = []
        for n in order:
            for c in colors:
                if c == X[n]: continue
                       
                diff, n_bc_c0, n_bc_c = op1(G, X, n, c, n_cc, n_bc)
                if diff >= 0: continue
                moves_1.append((n, c, diff, n_bc_c0, n_bc_c))
             
        if verbose:
            print '    --- moves_1', len(moves_1), [x[0] for x in moves_1]        
        if not moves_1:
            # Local minimum
            if verbose:
                print '    **** mininum', len(set(X)), best_cX_list[0][0]  
            add_best(nX)
           
            tested.add(hX)
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
    
    print 'best_cX_list', [c for c,_ in best_cX_list]
   
    return best_cX_list[0][1]                    

def repopulate(G, X0, Cso2):  
    print 'repopulate ------------------------'
    X = list(X0)
    color_counts = defaultdict(int)
    for c in X:
        color_counts[c] += 1
    assert sum(color_counts.values()) == len(X)    
    assert len(color_counts) == len(set(X))
    #colors = color_counts.keys()
    n_colors = len(color_counts)
    # excluded = set([c for c, k in color_counts.items() if k <= n_colors/2])
    max_color = max(color_counts.values())
    
    excluded1 = set([c for c, k in color_counts.items() if k <= 1]) 
    excluded2 = set([c for c, k in color_counts.items() if k <= 2])     
    print '** excluded1', len(excluded1), (len(color_counts), len(X)), max(color_counts.values())
    print '** excluded2', len(excluded2),
    print [(c,color_counts[c]) for c in sorted(color_counts, key=lambda x: -color_counts[x])]
    
    hX0 = hash(X0)
    for excluded in excluded1, excluded2:
        for _ in range(40):
            for i in range(len(X)):
                if color_counts[X[i]] in excluded or random.randrange(1, 3) == 1:
                    X[i] = -1
            X = populate2(G, X, Cso2)
            X = normalize(X)
            hX = hash(X)
            if hX != hX0:
                break
        if hX != hX0:
            break    
            
    print 'X0', hash(X0), X0
    # print 'X ', hash(X), X
    print 'X', hash(X), X
    print '---- repopulated'
    assert hX != hX0
    return X    

import random    
 
def solve(n_nodes, n_edges, edges):
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
    print 'n_min=%d' % (n_min)
    
    solutions = SortedDeque([], DEQU_LEN)
    visited = set([])
    visited_X = set([])
    X_list = []
    #X_list.append(populate1(G, X, Cso))
    #X_list.append(populate2(G, X, Cso))
    
    def doneX(X0):
        X = normalize(X0)
        hX = hash(X)
        #hashes = [hash(s[1]) for s in solutions]
        #print hX, hashes
        assert hX in hashes
        #assert hX not in visited
        if hX in visited:
            print 'doneX ------------------------'
            print visited
            for x in visited_X:
                print 'x ', hash(X), x
            print 'X0', hash(X0), X0
            print 'X ', hash(X),  X
            print '!!! duplicate'
            assert False
            return True
        visited.add(hX) 
        visited_X.add(X)    
        return False
        
    X = [-1] * len(G)
    X = populate1(G, X, Cso)    
    
    if False:
        Cso2 = Cso[:]
        for _ in range(10):
            random.shuffle(Cso2)
            X = populate1(G, X, Cso2)
            X = normalize(X)
            hX = hash(X)
            assert hX not in visited
            if hX in visited:
                continue
            visited.add(hX)    
            X_list.append(X)
        
        
        for X in X_list:
            X = normalize(X)
            add_solution(solutions, G, X)
    
    #n_colors = max(len(set(X1)), len(set(X2)))
    
    
    #add_solution(solutions, G, n_colors, X1)
    #add_solution(solutions, G, n_colors, X2)
    
    while len(solutions) < 10000:
        #X = do_kempe(G, X)
        X = normalize(X)
        #add_solution(solutions, G, X)
        optimal = len(set(set(X))) == n_min
        if not optimal:
            X = do_search(G, X)
            X = normalize(X)
            optimal = len(set(X)) == n_min
            print 'i=%d,optimal=%s' % (len(solutions), optimal)
            add_solution(solutions, G, X)
        if optimal:
            break
        print '------------'
        visited = [hash(s[1]) for s in solutions]
        print 'solutions', len(solutions), [(s[0],hash(s[1])) for s in solutions]
        print 'visited', len(visited), visited    
        X = repopulate(G, X, Cso)  
        hX = hash(X)
        print 'hX', hX
        if hX in visited:
            break
        
    exit()    
         
    for n in G:
        assert X[n] >= 0, '%d %s' % (n, G[n])
        assert all(X[n] != X[n1] for n1 in G[n]), '\n%d %s\n%d %s' % (
            n, G[n], X[n], [X[n1] for n1 in G[n]] ) 
    
    for n in G:
        print n, X[n], G[n], [X[i] for i in G[n]]
        
    print len(set(X)), optimal    
    print '+' * 40    
    return len(set(X)), X, optimal
    
    #print G    
    assert len(G) == n_nodes
    #print '-' * 80 
    
    # Nodes by order
    C = [-1] * n_nodes
    C[0] = 0
    
    Q, V = [0], set([])
    while Q:
        n = Q.pop()
        if n in V:
            continue
        V.add(n)
        #print n, G[n],
        for c in count():
         #   print c, 
            if not any(c == C[i] for i in G[n]):
                break
        C[n] = c        
        #print ':', c 
        for m in G[n]:
            Q.append(m)
        
  
    #print '-' * 80            
    #print C  
    #print '=' * 80  
    return len(set(C)), [C[i] for i in range(n_nodes)], True  

    


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

