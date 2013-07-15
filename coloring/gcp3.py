#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
from itertools import count
from collections import defaultdict
import pprint

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
    
    color_index = [-1] * (max(X) + 1)
    for i in set(X):
        color_index[i] = colors.index(i)
    #print colors, sorted(set(X)), color_index
    return tuple([color_index[c] for c in X])  
    
    
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

 
def populate(G, X, Cso): 
    # Color the graph somewhat like http://carajcy.blogspot.com.au/2013/01/dsatur.html
    # Start with largest clique
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
    return X                
    

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

    if False:
        #print (dict(G))
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
   
        
    
    # Lower and upper bounds on number of colors    
    n_min = len(Cso[0][0])
    n_max = len(G)
    
    X = [-1] * len(G)
    
    if False:
        # Color the graph somewhat like http://carajcy.blogspot.com.au/2013/01/dsatur.html
        # Start with largest clique
        cs, co = Cso[0]
        for i, n in enumerate(co):
            X[n] = i
        
        # Color the other cliques
        for cs, co in Cso[1:]:
            for n in co:
                if X[n] != -1:
                    continue
                neighbor_colors = set(X[i] for i in G[n])    
                for color in count():    
                    if color not in neighbor_colors:
                        X[n] = color
                        break
    else:
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
    
    if False:
    
        # Swap colors to reduce total number of colors
        while n_colors > n_min:
            last_color = n_colors - 1
            # nodes and cliques containing the color to be swapped
            nodes = [i for i, x in enumerate(X) if x == last_color]
            cliques = [cso for cso in Cso if any(n in nodes for n in cso[0])]
            
            # Might be easir to just sort by color count in reverse
            collision_counts = []
            for x in range(last_color):
                X2 = [x if y == last_color else y for y in X]
                collision_counts.append((x, count_collisions(G, X2)))
            collision_counts.sort(key=lambda x: -x[1])
            
            for cs, co in cliques:
                pass       
        
    return X, n_min                
 
def do_kempe(G, X0, n_colors):
    X = X0[:]
    
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
    print [len(cls) for cls in color_classes], objective()
     
    
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
                if False:
                    print 
                    for n in color_classes[i1]:
                        if i2 in G[n]:
                            print ' -- ', n, G[n]
                           
                    for n in color_classes[i2]:
                        if i1 in G[n]:
                            print ' ++ ', n, G[n]    
                    print        
                    print i1, sorted(set(sum((G[n] for n in color_classes[i1]), [])))
                    print i2, sorted(set(sum((G[n] for n in color_classes[i2]), [])))
                    print
                    print i1, len(color_classes[i1]), sorted(color_classes[i1])
                    print i2, len(color_classes[i2]), sorted(color_classes[i2])
                    print
                    print sorted(links1) 
                    print sorted(links2)
                    print
                    print sorted(color_classes[i1] | color_classes[i2])
                    print sorted(cc1)
                    print sorted(cc2)
                    print sorted(cc1 | cc2)
                if len(cc2) > len(cc1):
                    color_classes[i1] = cc2
                    color_classes[i2] = cc1
                else:    
                    color_classes[i1] = cc1
                    color_classes[i2] = cc2
                print objective(), (i1, i2), (before, after), after > before, [len(cls) for cls in color_classes]
                X = makeX()
                validate(G, X)    

    print '$' * 40
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

 
 
from utils import SortedDeque
    
def do_search(G, X, n_colors, Cso):
    """Local search around X using sum(|B[i]||C[i]| - |C[i]|^2) objective
    """
    
    for n in G:
        for n1 in G[n]:
            assert n in G[n1], '%d %d' % (n, n1)

    def makeX():  
        X1 = [-1] * len(X)
        for c, cls in enumerate(color_classes):
            for x in cls:
                X1[x] = c
        return X1 
    
    # color_classes[c] = nodes with color c
    color_classes = [set([n for n, x in enumerate(X) if x == c]) for c in range(n_colors)]
    assert sum(len(cls) for cls in color_classes) == len(X)
    
    # broken_classes[c] = edges where both colors are c
    broken_classes = [set([]) for c in range(n_colors)] 
    
    # We only need counts for our objective functions
    n_cc = [len(color_classes[c]) for c in range(n_colors)]
    n_bc = [len(broken_classes[c]) for c in range(n_colors)]
            
    def objective(n_cc, n_bc):
        return sum((2 * n_cc[c] * n_bc[c] - n_cc[c] ** 2) for c in range(n_colors))
    
    v = objective(n_cc, n_bc)  
    LEN = 10000
    solutions = SortedDeque([(v, normalize(X), n_cc, n_bc)], LEN)
    tested = set()
    counter = count()
    best_v = v
    best_X = tuple(X)
    best_n_col = len([x for x in n_cc if x > 0])
    
    normX = normalize(X)
    print 'best_X', v, best_n_col, color_counts(normX, n_colors), normX
    
    needs_perturbation = False
    
    while solutions: 
        v, X, n_cc, n_bc = solutions.popleft()
        check_counts(G, n_colors, X, n_cc, n_bc)
        #print '++++', v, ([solutions[i][0] for i in range(min(10,len(solutions)))], 
        #                  [solutions[-i-1][0] for i in range(min(10,len(solutions)))] )  
        if hash(normalize(X)) in tested:
            continue
        tested.add(hash(X))    
        
        #assert v == objective(n_cc, n_bc)
        n_col = len([x for x in n_cc if x > 0])
        print '*', (v, n_col), len(solutions), len(tested), (best_v, best_n_col) #, X, n_cc, n_bc
        if False:
            if len(tested) % 100 == 1:
                print X
                print n_cc
                print n_bc
                validate(G, X)
            
        if n_col < best_n_col or (n_col == best_n_col and v < best_v):
            if n_col < best_n_col:    
                normX = normalize(X)
                print 'best_X', v, n_col, color_counts(normX, n_colors), normX
                n_col_actual = validate(G, X)
                assert n_col == n_col_actual, 'n_col=%d n_col_actual=%d' % (n_col, n_col_actual)
            best_v = v
            best_X = X[:]
            best_n_col = n_col
            needs_perturbation = True

        
        if len(tested) % 10 == 1:
            needs_perturbation = True
            
        if needs_perturbation: #  and all(x == 0 for x in n_bc):
            needs_perturbation = False
            print 'perturbations',
            for X1 in perturb_by_class(G, X, Cso, n_colors):
                if hash(normalize(X1)) in tested:
                    continue
                n_cc1 = color_counts(X1, n_colors)
                n_bc1 = broken_counts(G, n_colors, X1)
                check_counts(G, n_colors, X1, n_cc1, n_bc1)
                v1 = objective(n_cc1, n_bc1)
                print v1, 
                solutions.insert((v1, X1, n_cc1, n_bc1))
            print    
        
        for n in G:
            c0 = X[n]
            for c in range(n_colors):
                # Looking for a new color
                if c == c0:
                    continue
                #X2[n] = c
                #d = delta(n, c, n_cc[c0], n_cc[c], n_bc[c0], n_bc[c])
                c0 = X[n]
                # Can't remove any c0 color nodes if none exist
                if n_cc[c0] == 0:
                    continue
                n_cc_c0, n_cc_c, n_bc_c0, n_bc_c = n_cc[c0], n_cc[c], n_bc[c0], n_bc[c]
                before = 2 * n_cc_c0 * n_bc_c0 - n_cc_c0**2 + 2 * n_cc_c * n_bc_c - n_cc_c**2
                n_cc_c0 -= 1
                n_cc_c += 1
                n_bc_c0 -= sum(int(X[n1] == c0) for n1 in G[n])
                n_bc_c  += sum(int(X[n1] == c)  for n1 in G[n])
                after = 2 * n_cc_c0 * n_bc_c0 - n_cc_c0**2 + 2 * n_cc_c * n_bc_c - n_cc_c**2
                #print '&', before, after
                if after < before: #and v + after - before >= best_v:
                    X2, n_cc2, n_bc2 = list(X), n_cc[:], n_bc[:]
                    X2[n] = c
                    n_cc2[c0], n_cc2[c], n_bc2[c0], n_bc2[c] =  n_cc_c0, n_cc_c, n_bc_c0, n_bc_c
                    check_counts(G, n_colors, X2, n_cc2, n_bc2)
                    v2 = objective(n_cc2, n_bc2)
                    #assert v2 == v + after - before
                    if hash(normalize(X2)) in tested:
                        continue
                    solutions.insert((v + after - before, tuple(X2), n_cc2, n_bc2))

    return best_X                    
                    
 
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
    
    X, n_min = find_min_colors(G, Cso) 
    n_colors = len(set(X))
    optimal = n_colors == n_min
    print 'n_min=%d,n_colors=%d' % (n_min, n_colors)
    
    if not optimal:
        X = do_kempe(G, X, n_colors)
        X = normalize(X)
        n_colors = len(set(X))
        optimal = n_colors == n_min
       
    if True:    
        if not optimal:
            X = do_search(G, X, n_colors, Cso)
            X = normalize(X)
            n_colors = len(set(X))
            optimal = n_colors == n_min
         
    for n in G:
        assert X[n] >= 0, '%d %s' % (n, G[n])
        assert all(X[n] != X[n1] for n1 in G[n]), '\n%d %s\n%d %s' % (
            n, G[n], X[n], [X[n1] for n1 in G[n]] ) 
    
    for n in G:
        print n, X[n], G[n], [X[i] for i in G[n]]
        
    print n_colors, optimal    
    print '+' * 40    
    return n_colors, X, optimal
    
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

