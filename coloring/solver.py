#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
from itertools import count
from collections import defaultdict

def xx(lst):
    return [sorted(x) for x in lst]
    

def clique(G, n0):
    """Return maximal clique in graph G containing node n0"""
    # Q = a stack
    # V = visited nodes
    Q, V = [n0], set([n0])
    
    
    while Q:
        #print '++', Q, list(V),
        n = Q.pop()
        #print n
        for n1 in G[n]:
            # Already visited?
            if n1 in V:
                continue
            # Part of clique ?    
            if not all(n2 in G[n1] for n2 in V):
                continue
            V.add(n1)
            Q.append(n)
           # print '--', n1, Q, list(V)

    return frozenset(V)


def all_cliques(G):
    """Return all maximal cliques in G"""
    # N^2 
    C = [clique(G, n) for n in G]
    
    print '**        maximal cliques' , xx(C)
    #C_unique = sorted(set(C), key=lambda c: C.index(c))
    C_unique = sorted(set(C), key=lambda c: (-len(c), C.index(c)))
    
    print '!! unique maximal cliques' , xx(C_unique)
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
        print i, xx(C), c, c_all, ll
    
    print '-' * 80
    return C
 

def find_min_colors(G):

    print G
    order = range(len(G))
    order.sort(key=lambda n: -len(G[n]))
    # Cliques as sets
    Cs = all_cliques(G)
    
    
    # Cliques as ordered lists
    Co = [sorted(c, key=lambda n: -len(G[n])) for c in Cs]
    print '>>', Co
    Cso = zip(Cs, Co)
    
    n_min = len(Cso[0])
    n_max = len(G)
    
    X = [-1] * len(G)
    
    cs, co = Cso[0]
    for i, n in enumerate(co):
        X[n] = i
    
    for cs, co in Cso[1:]:
        for n in co:
            if X[n] != -1:
                continue
            neighbor_colors = set(X[i] for i in G[n])    
            for color in count():    
                if color not in neighbor_colors:
                    X[n] = color
                    break
                    
    return len(set(X)), X, False                
        

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

