#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
  x[w] = 1 if warehouse w is open else 0
  y[w,c] = 1 if customer c uses warehouse w else 0
  obj = sum([warehouses[w].cost * x[w]) fixed cost  
      + sum(customerCosts[w,c] *y[w,c]) transportation cost
    
Entire model:
    COST = obj
    
    x[w] = 0,1
    y[w,c] = 0,1
    y[w,c] <= x[w]
    sum(y[w,c]) over w == 1 
    sum(y[w,c]) over c <= capacity[w]

"""
from __future__ import division
from collections import namedtuple
import os

Warehouse = namedtuple('Warehouse', ['capacity', 'cost'])
Row = namedtuple('Row', ['name', 'typ', 'vals', 'rhs'])

MPS_DIR = 'mps'

try:
    os.makedirs(MPS_DIR)
except:
    pass
    
def make_name(W, C):
    return 'wh_%03d_%05d.mps' % (W, C)
    
    
def make_path(W, C):
    return os.path.join(MPS_DIR, make_name(W, C))
    

def make_mps(warehouses, customerCount, customerSizes, customerCosts):
    
    W = len(warehouses)
    C = customerCount
    print 'make_mps', W, C
    
    x_vars = ['x%d' % w for w in range(W)]
    y_vars = ['y%d_%d' % (w, c)  for w in range(W)
                    for c in range(C)]
    variables = x_vars + y_vars

    rows = []
    
    def make_row(cls, row_type, x_rule, y_rule, rhs):
        x = [0] * W
        y = [0] * C * W
        for w in range(W):
            x[w] = x_rule(w)
        for w in range(W):
            for c in range(C):
                #print c, w, c * W + w, C * W
                y[w * C + c] = y_rule(w, c)
        name = '%s%02d' % (cls, len(rows))        
        #print cls, len(rows), name, rhs
        rows.append(Row(name=name, typ=row_type, vals=x+y, rhs=rhs))
      
    print '@1'  
    #obj = sum([warehouses[w].cost * x[w]) fixed cost  
    #  + sum(customerCosts[w,c] *y[w,c]) transportation cost
    make_row('OB', 'N', 
            lambda i: warehouses[i].cost, 
            lambda i, j: customerCosts[j][i], 0.0)
    
    # y[w,c] <= x[w]
    for w in range(W):
        for c in range(C):
            make_row('A_', 'G', lambda i: int(i==w), lambda i, j: -int(i==w and j==c), 0.0)
    
    print '@2'
    #sum(y[w,c]) over w == 1     
    for c in range(C):
        make_row('B_', 'E', lambda i: 0, lambda i, j: int(j==c), 1.0)   
    
    print '@3'
    # sum(y[w,c]) over c <= capacity[w]
    for w in range(W):
        print 'warehouses[%d]=%s' % (w, warehouses[w])
        make_row('C_', 'L', lambda i: 0, lambda i, j: int(i==w) , warehouses[w].capacity)    
        
    print '@100', len(rows)
   
    
    with open(make_path(W,C), 'wt') as f:
        f.write('*NAME:         %s\n' % make_name(W, C))
        f.write('*ROWS:         %d\n' % len(variables))
        f.write('*COLUMNS:      %d\n' % len(rows))
        for r in rows:
            f.write('* %s\n' % repr(r)[:100])
        #f.write('*INTEGER:      27
        f.write('%-15s%s\n' % ('NAME', make_name(W, C)))
        f.write('ROWS\n')
        #f.write('%2s %s\n' % ('N', 'OBJ'))
        for r in rows:
            f.write('%2s %s\n' % (r.typ, r.name))
        f.write('COLUMNS\n')  
        f.write("  MARK0000  'MARKER'                 'INTORG'\n")
        for iv, v in enumerate(variables):
            for ir, r in enumerate(rows):
                #print ir, r
                if ir % 2 == 0:
                     f.write('%7s ' % v)
                f.write('%7s %8.3f ' % (r.name, r.vals[iv]))
                if ir % 2 == 1:
                     f.write('\n')
            f.write('\n')             
        f.write("   MARK0001  'MARKER'                 'INTEND'\n")
        f.write('RHS\n')             
        for ir, r in enumerate(rows):
            f.write(' %7s %8.3f' % (r.name, r.rhs))
            if ir % 2 == 1:
                 f.write('\n') 
        f.write('\n')      
        f.write('BOUNDS\n')          
        for iv, v in enumerate(variables):
           # f.write('  LO BOUND %7s 0\n' % (v))
            f.write('  UP BOUND %7s 1\n' % (v))
        f.write('ENDATA\n')     

# build a trivial solution
# pack the warehouses one by one until all the customers are served

def solve(warehouses, customerCount, customerSizes, customerCosts):
    
    make_mps(warehouses, customerCount, customerSizes, customerCosts)
    exit()
    if False:
        solution = [-1] * customerCount
        capacityRemaining = [w.capacity for w in warehouses]

        warehouseIndex = 0
        for c in range(customerCount):
            if capacityRemaining[warehouseIndex] >= customerSizes[c]:
                solution[c] = warehouseIndex
                capacityRemaining[warehouseIndex] -= customerSizes[c]
            else:
                warehouseIndex += 1
                assert capacityRemaining[warehouseIndex] >= customerSizes[c]
                solution[c] = warehouseIndex
                capacityRemaining[warehouseIndex] -= customerSizes[c]
        return solution

def solveIt(inputData):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = inputData.split('\n')

    parts = lines[0].split()
    warehouseCount = int(parts[0])
    customerCount = int(parts[1])

    warehouses = []
    for i in range(1, warehouseCount+1):
        line = lines[i]
        parts = line.split()
        warehouses.append(Warehouse(capacity=int(parts[0]), cost=float(parts[1])) )

    customerSizes = []
    customerCosts = []

    lineIndex = warehouseCount + 1
    for i in range(customerCount):
        customerSize = int(lines[lineIndex+2*i])
        customerCost = map(float, lines[lineIndex+2*i+1].split())
        customerSizes.append(customerSize)
        customerCosts.append(customerCost)
            
        
                
    solution = solve(warehouses, customerCount, customerSizes, customerCosts)              

    used = [0]*warehouseCount
    for wa in solution:
        used[wa] = 1

    # calculate the cost of the solution
    obj = sum([warehouses[x].cost * used[x] for x in range(warehouseCount)])
    for c in range(customerCount):
        obj += customerCosts[c][solution[c]]

    # prepare the solution in the specified output format
    outputData = str(obj) + ' ' + str(0) + '\n'
    outputData += ' '.join(map(str, solution))

    return outputData


import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        fileLocation = sys.argv[1].strip()
        inputDataFile = open(fileLocation, 'r')
        inputData = ''.join(inputDataFile.readlines())
        inputDataFile.close()
        print 'Solving:', fileLocation
        print solveIt(inputData)
    else:
        print 'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/wl_16_1)'

