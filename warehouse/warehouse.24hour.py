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
import os, glob, subprocess, re
import logging

Warehouse = namedtuple('Warehouse', ['capacity', 'cost'])
Row = namedtuple('Row', ['name', 'typ', 'vals', 'rhs'])

MPS_DIR = 'mps'
SOLUTION_DIR = 'solution'
BAT_DIR = 'batch'
SCIP_PATH = r'D:\peter.dev\disc.op\scip-3.0.1.mingw.x86_64.intel.opt.spx.exe\scip.exe' 
#SCIP_PATH = r'D:\dev\coursera\discrete.optimization\scip-3.0.1.mingw.x86_64.intel.opt.spx.exe\scip.exe'


for dir in MPS_DIR, SOLUTION_DIR, BAT_DIR:
    try:
        os.makedirs(dir)
    except:
        pass
    
def make_name(path):
    return os.path.basename(path)
    
def make_mps_path(path):
    return os.path.join(MPS_DIR, make_name(path) + '.mps')

def make_solution_path(path):
    return os.path.join(SOLUTION_DIR, make_name(path) + '.24hour.soln')    
       
def make_bat_path(path):
    return os.path.join(BAT_DIR, make_name(path) + '.scip')    
    
def run_scip(bat_path):    
    argv = [SCIP_PATH, '-b', bat_path, '-s', 'scip.24hr.set' ]
    p = subprocess.Popen(argv)
    p.wait()
    

RE_OBJ = re.compile(r'\d+\.\d+')    
RE_Y = re.compile(r'y(\d+)_(\d+)') 
 
def parse_solution(solution_path):
    optimal = False
    objective = None
    solution_dict = {}
    with open(solution_path, 'rt') as f:
        for line in f:
            if 'solution status' in line:
                optimal = 'optimal' in line
            elif  'objective value' in line: 
                m = RE_OBJ.search(line)
                objective = float(m.group(0))
            else:
                m = RE_Y.search(line)
                if m:
                    w = int(m.group(1))
                    c = int(m.group(2))
                    assert c not in solution_dict.keys(), c
                    solution_dict[c] = w
               
    return objective, optimal, solution_dict
    

def make_mps(path, warehouses, customerCount, customerSizes, customerCosts):
    
    W = len(warehouses)
    C = customerCount
    print 'make_mps', W, C
    
    x_vars = ['x%d' % w for w in range(W)]
    y_vars = ['y%d_%d' % (w, c)  for w in range(W)
                    for c in range(C)]
    variables = x_vars + y_vars

    rows = []
    # For looking up rows which contain a particular variable
    rows_with_var = {v:set() for v in variables}
    
    def make_row(cls, row_type, 
                 x_rule, x_w_keys, 
                 y_rule, y_w_keys, y_c_keys, 
                 rhs):
        vals = {}
        for w in x_w_keys:
            vals['x%d' % w] = x_rule(w)
        for w in y_w_keys:
            for c in y_c_keys:
                #print c, w, c * W + w, C * W
                vals['y%d_%d' % (w, c)] = y_rule(w, c)
  
        num = len(rows)
        name = '%s%02d' % (cls, num)        
        #print cls, len(rows), name, rhs
        rows.append(Row(name=name, typ=row_type, vals=vals, rhs=rhs))
        
        for v in vals:
            rows_with_var[v].add(num) 
      
    if not os.path.exists(make_mps_path(path)):  
        print '@1 objective'   
        #obj = sum([warehouses[w].cost * x[w]) fixed cost  
        #  + sum(customerCosts[w,c] *y[w,c]) transportation cost
        make_row('OB', 'N', 
                lambda i: warehouses[i].cost, range(W),
                lambda i, j: customerCosts[j][i], range(W), range(C),
                0.0)
        
        print '@1a', W * C
        # y[w,c] <= x[w]
        for w in range(W):
            for c in range(C):
                make_row('A_', 'G', 
                    lambda i: 1, [w],
                    lambda i, j: -1, [w], [c],  
                    0.0)
        
        print '@2', C
        #sum(y[w,c]) over w == 1     
        for c in range(C):
            make_row('B_', 'E', 
                lambda i: 0, [],
                lambda i, j: 1, range(W), [c], 
                1.0)   
        
        print '@3', W
        # sum(y[w,c]) over c <= capacity[w]
        for w in range(W):
            #print 'warehouses[%d]=%s' % (w, warehouses[w])
            make_row('C_', 'L', 
                lambda i: 0, [],
                lambda i, j:  customerSizes[j], [w], range(C), 
                warehouses[w].capacity)    
            
        print '@100', make_mps_path(path), len(rows)
        
        for r in rows:
            assert isinstance(r, Row), r
        
        for v in rows_with_var.keys():
            rows_with_var[v] = sorted(rows_with_var[v])
            assert isinstance(rows_with_var[v], list), rows_with_var[v]
       
        with open(make_mps_path(path), 'wt') as f:
            f.write('*NAME:         %s\n' % make_name(path))
            f.write('*ORIG NAME:    %s\n' % path)
            f.write('*W:            %d\n' % W)
            f.write('*C:            %d\n' % C)
            f.write('*COLUMNS:      %d\n' % len(rows))
            #for r in rows:
            #    f.write('* %s\n' % repr(r))
            #f.write('*INTEGER:      27
            f.write('%-15s%s\n' % ('NAME', make_name(path)))
            f.write('ROWS\n')
            #f.write('%2s %s\n' % ('N', 'OBJ'))
            for r in rows:
                f.write('%2s %s\n' % (r.typ, r.name))
            f.write('COLUMNS\n')  
            f.write("  MARK0000  'MARKER'                 'INTORG'\n")
            total = 0
            for iv, v in enumerate(variables):
                if iv % 100000 == 10000:
                    estimate = int(total * len(variables)/iv) if iv > 0 else 0
                    print '%6d of %d %f (%6d) : %6d' % (iv, len(variables), iv/len(variables), total, estimate)
                    f.flush()
                cnt = 0
        
                for ir in rows_with_var[v]: 
                    r = rows[ir]
                #for ir, r in enumerate(rows):
                #    if v not in r.vals.keys(): 
                #        continue
                    if cnt % 2 == 0:
                        f.write(' %-7s ' % v)
                    f.write('%-7s %8.3f ' % (r.name, r.vals.get(v, 0)))
                    if cnt % 2 == 1:
                        f.write('\n')
                    cnt += 1 
                    total += 1    
                f.write('\n')             
            f.write("   MARK0001  'MARKER'                 'INTEND'\n")
            f.write('RHS\n')             
            for ir, r in enumerate(rows):
                f.write(' %-7s %8.3f' % (r.name, r.rhs))
                if ir % 2 == 1:
                     f.write('\n') 
            f.write('\n')      
            f.write('BOUNDS\n')          
            for iv, v in enumerate(variables):
               # f.write('  LO BOUND %-7s 0\n' % (v))
                f.write('  UP BOUND %-7s 1\n' % (v))
            f.write('ENDATA\n')     
            

    with open(make_bat_path(path), 'wt') as f:
        f.write('read %s\n' % make_mps_path(path))
        f.write('optimize\n')
        f.write('write solution %s\n' % make_solution_path(path))
        f.write('quit\n')

    run_scip(make_bat_path(path))
    objective, optimal, solution_dict = parse_solution(make_solution_path(path))   
    assert len(solution_dict) == C
    
    solution = [-1] * C
    for c, w in solution_dict.items():        
        solution[c] = w    
        
    print objective, optimal, solution 
    print '*' * 80
    
    logging.info(' %s, W=%d, C=%d, %s, %s, %s' % (path, W, C, objective, optimal, solution))


# build a trivial solution
# pack the warehouses one by one until all the customers are served

def solve(path, warehouses, customerCount, customerSizes, customerCosts):
    
    make_mps(path, warehouses, customerCount, customerSizes, customerCosts)
   
# The problems that are assessed in the course (from submit.pyc)   
some_dict = {    
 '8zVuNh0L-dev': './data/wl_16_1',
 'eaOs8z3l-dev': './data/wl_25_2',
 'pxIsuCQe-dev': './data/wl_50_1',
 'clCLq0Vb-dev': './data/wl_100_4',
 '2gHetJOV-dev': './data/wl_200_1',
 'ASSX0Lq4-dev': './data/wl_500_1',
 'OXs89Owz-dev': './data/wl_1000_1',
 'AO0FA8y8-dev': './data/wl_2000_1'    
}
submit_paths = some_dict.values()

RE_DATA = re.compile(r'wl_(\d+)_\d+') 

def get_W(path):
    """Return number of warehouses (W) for data file in path
        e.g. path='data/wl_100_4' => W=100
    """    
    m = RE_DATA.search(path)
    assert m, path
    return int(m.group(1))
    
w_path_map = { get_W(path): path for path in submit_paths }
submit_paths = [w_path_map[w] for w in sorted(w_path_map)]



def solve_extern(warehouses, customerCount, customerSizes, customerCosts):
    W = len(warehouses)
    C = customerCount
    print 'solve_extern', W, C
    path = w_path_map[W]
    
    solution_path = make_solution_path(path)
    objective, optimal, solution_dict = parse_solution(solution_path)
    
    assert len(solution_dict) == C
    solution = [-1] * C
    for c, w in solution_dict.items():        
        solution[c] = w  
        
    return objective, optimal, solution   
    
       

def solveIt(path, inputData):
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
            
    solve(path, warehouses, customerCount, customerSizes, customerCosts)        
    return             
    #solution = solve(path, warehouses, customerCount, customerSizes, customerCosts)              

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

    
    logging.basicConfig(filename='log.log', level=logging.DEBUG, 
            format='%(asctime)s:%(levelname)s:%(message)s')

    logging.info('-' * 80)
    logging.info('Starting')

    path_list = list(glob.glob('data/*'))
    path_list.sort(key=lambda path: os.path.getsize(path))
    path_list = submit_paths
    
    
    for i, path in enumerate(path_list):
        print i, path
    print '-' * 80
   
    #for path in reversed(path_list):
    for path in path_list[5:]:
        with open(path, 'r') as f:
            inputData = ''.join(f.readlines())
        print 'Solving:', path
        solveIt(path, inputData)

    if len(sys.argv) > 1:
        fileLocation = sys.argv[1].strip()
        inputDataFile = open(fileLocation, 'r')
        inputData = ''.join(inputDataFile.readlines())
        inputDataFile.close()
        print 'Solving:', fileLocation
        print solveIt(inputData)
    else:
        print 'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/wl_16_1)'

