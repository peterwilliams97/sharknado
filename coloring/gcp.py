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
    
    return [frozenset(c) for c in (C_maximal + C)]       
                    
    
    Q, V = [n0], set([n0])
    
    while Q:
        #print '++', Q, list(V),
        n = Q.pop()
        #print n
        for n1 in G[n]:
            # Already visited?
            for v in V:
                if n1 not in V:
                    # Part of clique ?    
                    if all(n2 in G[n1] for n2 in V):
                        V.add(n1)
                        Q.append(n1)
           # print '--', n1, Q, list(V)

    return frozenset(V)


def all_cliques(G):
    """Return all maximal cliques in G"""
    # N^2 
    C = []
    for n in G:
        C.extend(maximal_cliques(G, n))
    #C = [clique(G, n) for n in G]
    
    print '**        maximal cliques' 
    for i, c in enumerate(C):
        print '%2d : %s' % (i, sorted(c))
    #C_unique = sorted(set(C), key=lambda c: C.index(c))
    C_unique = sorted(set(C), key=lambda c: (-len(c), C.index(c)))
    
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
    
    C = [C_unique[0]]
    c_all = set(C_unique[0])
    print xx(C), c_all
    for i in range(1, len(C_unique)):
        c = max(C_unique[i:], key=lambda c: -len(c & c_all))
        ll = len(c & c_all)
        C.append(c)
        c_all.update(c)
        #print i, xx(C), c, c_all, ll
    
    print '-' * 80
    return C
 
 
def count_collisions(G, X):
    n_collisions = 0
    for n in G:
        for n1 in G[n]:
            if X[n1] == X[n]:
                n_collisions += 1
    return n_collisions
    
    
def find_min_colors(G):

    pp(dict(G))
    order = range(len(G))
    order.sort(key=lambda n: -len(G[n]))
    # Cliques as sets
    Cs = all_cliques(G)
    
    
    # Cliques as ordered lists
    Co = [sorted(c, key=lambda n: -len(G[n])) for c in Cs]
    print '>>', Co
    Cso = zip(Cs, Co)
    
    # Lower and upper bounds on number of colors    
    n_min = len(Cs[0])
    n_max = len(G)
    
    X = [-1] * len(G)
    
    # Color the graph  Kind of like http://carajcy.blogspot.com.au/2013/01/dsatur.html
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
                    
    n_colors = len(set(X))
    print n_min, n_colors, n_max
    
    for c in sorted(set(X)):
        print '%d: %d' % (c, X.count(c))
    
    # Sort colors by most common first
    color_order = sorted(set(X), key=lambda c: (-X.count(c), c))
    color_order = [color_order.index(i) for i in range(n_colors)]
    X = [color_order[c] for c in X]  
    print color_order
    
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
        
    return n_colors, X, n_colors == n_min                
 
def do_kempe(G, X, n_colors):
    color_classes = [set([n for n, x in enumerate(X) if x == c]) for c in range(n_colors)]
    assert sum(len(cls) for cls in color_classes) == len(X)
    
    def objective():
        return sum(len(cls)**2 for cls in color_classes)
    
    print '$' * 40
    print [len(cls) for cls in color_classes], objective()
    for i1 in range(1, n_colors):
        for i2 in range(i1):
            links1 = set([n for n in color_classes[i1] if i2 in G[n]])
            links2 = set([n for n in color_classes[i2] if i1 in G[n]])
            cc1 = (color_classes[i1] - links1) | links2
            cc2 = (color_classes[i2] - links2) | links1
            before = len(color_classes[i1])**2 + len(color_classes[i2])**2
            after = len(cc1)**2 + len(cc2)**2
            if after > before:
                color_classes[i1] = cc1
                color_classes[i2] = cc2
                print [len(cls) for cls in color_classes], objective(), (i1, i2), (before, after), after > before 

    print '$' * 40          
 
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
    
    n_colors, X, optimal = find_min_colors(G) 
 
    if not optimal:
        do_kempe(G, X, n_colors)
    
    for n in G:
        assert X[n] >= 0, '%d %s' % (n, G[n])
        assert all(X[n] != X[n1] for n1 in G[n]), '%d %s' % (n, G[n]) 
    
    for n in G:
        print n, X[n], G[n], [X[i] for i in G[n]]
        
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
for path in glob.glob(mask):
    #print '-' * 80
    print path 
    with open(path, 'rt') as f:
        data = f.read()
    print solveIt(data)
   
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

