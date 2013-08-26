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
import math, copy
from collections import namedtuple
#import tsp4aj_best40_10_nonrandom_f as tsp
#import tsp4aj_best20_100_random_a as tsp
import tsp4aj_local50b as tsp

VERSION = 1
RANDOM_SEED = 111 # The Nelson!
MAX_EDGES = 100 # 2000

print 'VERSION=%d' % VERSION

random.seed(RANDOM_SEED)

Customer = namedtuple('Customer', ['demand', 'x', 'y'])

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

    
CACHE_DIR = 'cache.vrp'    
#hack_locations = None 
  
def precalculate(points): 
   
    N = len(points)
    #print '@@1'
    
    base_dir = os.path.join(CACHE_DIR, 'obs%05d' % N)
    npa_dict = { 'locations': None,   'distances': None, 'closest': None, 'locations_r' : None }
    #print '@@2', base_dir
    
    existing = False # !@#$
    #existing = load_save_npa(base_dir, npa_dict, False)
    if existing:
        print 'loading existing !!!!!!!!!!!!!!!!!!'
        print npa_dict.keys()
        locations, distances, closest = npa_dict['locations'], npa_dict['distances'], npa_dict['closest'] 
        locations_r = npa_dict['locations_r']
        
        print 'locations:', locations.shape
        print 'distances:', distances.shape
        print 'closest:', closest.shape

     
        assert locations.shape[0] == N
        assert distances.shape[0] == N
        assert distances.shape[1] == N
        assert closest.shape[0] == N
        assert closest.shape[1] == N - 1
    else:
    
        locations = np.empty((N, 2), dtype=np.float64)
        locations_r = np.empty((N, 2), dtype=np.float64)
        distances = np.empty((N, N), dtype=np.float64)  # !@#$%
        closest = np.empty((N, N-1), dtype=np.int32)
        row = np.empty(N, dtype=np.float64)
        
        x0, y0 = points[0].x, points[0].y
        for i in xrange(N):
            locations[i][0] = points[i].x - x0
            locations[i][1] = points[i].y - y0
            locations_r[i][0] = np.sqrt(locations[i][0] ** 2 + locations[i][1] ** 2)
            locations_r[i][1] = np.arctan2(locations[i][1], locations[i][0])
        
        #print '@@2a'
        #print 'locations:', locations.shape
        #print 'distances:', distances.shape
        #print 'closest:', closest.shape
        
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
 
            #print '@@4'    
        
        npa_dict['locations'], npa_dict['distances'], npa_dict['closest'] = locations, distances, closest
        npa_dict['locations_r'] = locations_r 
        load_save_npa(base_dir, npa_dict, True)
    
    #print '@@5'    
    return N, locations, locations_r, distances, closest 


def length(customer1, customer2):
    return math.sqrt((customer1.x - customer2.x)**2 + (customer1.y - customer2.y)**2)


def tour_dist(customers, depotIndex, tour):
    obj = length(customers[depotIndex], customers[tour[0]])
    for i in range(0, len(tour) - 1):
        obj += length(customers[tour[i]], customers[tour[i + 1]])
    obj += length(customers[tour[-1]], customers[depotIndex]) 
    return obj
    
def total_dist(customers, depotIndex, vehicleTours):
    return sum(tour_dist(customers, depotIndex, tour) for tour in vehicleTours if tour)    

    
def get_shortest_path(file_path, customers, depotIndex, tour): 
    if not tour:
        return tour
    dist0 = tour_dist(customers, depotIndex, tour)
    full_tour = [depotIndex] + tour
    points = [(customers[i].x, customers[i].y) for i in full_tour]
    #print [customers[i] for i in full_tour]
    #print points
    #exit()
    dist, order = tsp.solve(file_path, points)
    
    assert order[0] == 0
    
    #print points
    #print dist0, dist
    #print full_tour
    #print order
    shortest_tour = [full_tour[i] for i in order]
    #dist = tour_dist(customers, depotIndex, shortest_tour[1:])
    #print shortest_tour
    #if dist > dist0 + 1e-6:
    #    return tour
    assert dist <= dist0 + 1e-6, '%f %f %f' % (dist, dist0, dist0 - dist)
    return shortest_tour[1:]
    

def get_shortest_paths(file_path, customers, depotIndex, vehicleTours): 
    return [get_shortest_path(file_path, customers, depotIndex, tour)
            for tour in vehicleTours]

    
def solve0(customerCount, vehicleCount, vehicleCapacity, depotIndex, customers):   
    
    # build a trivial solution
    # assign customers to vehicles starting by the largest customer demands
    
    vehicleTours = []
    customerIndexs = set(range(1, customerCount))  # start at 1 to remove depot index

    for v in range(0, vehicleCount):
        # print "Start Vehicle: ",v
        vehicleTours.append([])
        capacityRemaining = vehicleCapacity
        while any(capacityRemaining >= customers[ci][0] for ci in customerIndexs):
            used = set()
            order = sorted(customerIndexs, key=lambda ci: -customers[ci][0])
            for ci in order:
                if capacityRemaining >= customers[ci][0]:
                    capacityRemaining -= customers[ci][0]
                    vehicleTours[v].append(ci)
                    # print '   add', ci, capacityRemaining
                    used.add(ci)
            customerIndexs -= used 
    
    return vehicleTours
    
   
 
def check(customerCount, customers, vehicleCapacity, vehicleTours):
    all_customers = []
    for tour in vehicleTours:
        all_customers.extend(tour)
    all_customers_set = set(all_customers)
    assert len(all_customers_set) == len(all_customers)
    for c in range(1, customerCount):
        assert c in all_customers_set, c
    for tour in vehicleTours:
        demand = sum(customers[c].demand for c in tour)
        assert demand <= vehicleCapacity
        

def is_valid(customerCount, customers, vehicleCapacity, vehicleTours):
    all_customers = []
    for tour in vehicleTours:
        all_customers.extend(tour)
    all_customers_set = set(all_customers)
    if len(all_customers_set) != len(all_customers):
        return False
    for c in range(1, customerCount):
        if not c in all_customers_set:
            return False
    for tour in vehicleTours:
        demand = sum(customers[c].demand for c in tour)
        if demand > vehicleCapacity:
            return False
    return True
        
def vehicles_for_order(customerCount, customers, vehicleCapacity, order):

    vehicleTours = []
    customerIndexes = set(range(1, customerCount))  # start at 1 to remove depot index
 
    capacityRemaining = vehicleCapacity
    tour = []
    for c in order:
        customer = customers[c]
        #print c, capacityRemaining, customer, locations_r[c, :],
        if customer.demand <= capacityRemaining:
            tour.append(c)
            capacityRemaining -= customer.demand 
            #print 'A'
        elif tour:
            #print '***', tour
            #assert len(vehicleTours) < vehicleCount
            vehicleTours.append(tour)
            tour = [c]
            capacityRemaining = vehicleCapacity - customer.demand 
            
    if tour:
        #print '!**', tour
        #assert len(vehicleTours) < vehicleCount
        vehicleTours.append(tour) 

    check(customerCount, customers, vehicleCapacity, vehicleTours)
    return vehicleTours
    
    
def fix_tours_(customers, vehicleCapacity, vehicleCount, vehicleTours):
    capacities = []
    for tour in vehicleTours:
        capacities.append(vehicleCapacity - sum(customers[c].demand for c in tour))
    t_min = min(range(len(vehicleTours)), key=lambda t: len(vehicleTours[t]))
    tour_min = vehicleTours[t_min][:]
    # print 'tour_min', tour_min
    for i, c in enumerate(tour_min):
        for t in range(len(vehicleTours)):
            if t == t_min: continue
            if c not in vehicleTours[t_min]: continue
            if capacities[t] >= customers[c].demand:
                vehicleTours[t].append(c)
                vehicleTours[t_min].remove(c)
                capacities[t] -= customers[c].demand
                # print '>', vehicleTours[t_min]
        if len(vehicleTours[t_min]) == 0:
            del vehicleTours[t_min]
            break  
            

def fix_tours(customers, vehicleCapacity, vehicleCount, vehicleTours):
    v0 = len(vehicleTours)
    while len(vehicleTours) > vehicleCount:
        fix_tours_(customers, vehicleCapacity, vehicleCount, vehicleTours)
        if len(vehicleTours) == v0:
            break
        v0 = len(vehicleTours)
        
 
def adjust_tours(customers, vehicleCapacity, vehicleCount, vehicleTours): 
    remaining = set(i for i in range(len(vehicleTours)) if vehicleTours[i])
    if len(remaining) < 2:
        return
    #print 'adjust_tours', remaining,
    vo = random.randrange(0, len(remaining))
    remaining.remove(vo)
    #print remaining,
    vi = random.randrange(0, len(remaining))
    #print vehicleTours[vo]
    c =  vehicleTours[vo][random.randrange(0, len(vehicleTours[vo]))]
    vehicleTours[vo].remove(c)
    vehicleTours[vi].append(c)
    fix_tours(customers, vehicleCapacity, vehicleCount, vehicleTours)
    

def best_order(customerCount, customers, vehicleCount, vehicleCapacity, angle_order):
    best_tours = None
    
    for i in range(len(angle_order)):
        order = angle_order[i:] + angle_order[:i]
        vehicleTours = vehicles_for_order(customerCount, customers, vehicleCapacity, order)
        fix_tours(customers, vehicleCapacity, vehicleCount, vehicleTours)
        if len(vehicleTours) <= vehicleCount:
            if best_tours is None or len(vehicleTours) < len(best_tours):
                best_tours = vehicleTours
            
    if not best_tours: return None
    check(customerCount, customers, vehicleCapacity, best_tours)
    return best_tours
    
    
def solve(customerCount, vehicleCount, vehicleCapacity, depotIndex, customers):
    """Return traversal order of points that minimizes distance travelled"""
        
    N, locations, locations_r, distances, closest = precalculate(customers)
   
    #print locations
    #print locations_r
    angle_order = range(1, N)
    angle_order.sort(key=lambda i: (locations_r[i, 1], locations_r[i, 0])) 
        
    vehicleTours = best_order(customerCount, customers, vehicleCount, vehicleCapacity, angle_order)
    if not vehicleTours:
        vehicleTours = solve0(customerCount, vehicleCount, vehicleCapacity, depotIndex, customers)
    check(customerCount, customers, vehicleCapacity, vehicleTours)
    vehicleTours = get_shortest_paths('file_path XXX', customers, depotIndex, vehicleTours)
    check(customerCount, customers, vehicleCapacity, vehicleTours)
    
    vehicleTours0 = copy.deepcopy(vehicleTours)
    dist0 = total_dist(customers, depotIndex, vehicleTours)
    if False:
        for _ in range(100):
            vehicleTours = copy.deepcopy(vehicleTours0) 
            adjust_tours(customers, vehicleCapacity, vehicleCount, vehicleTours)
            vehicleTours = get_shortest_paths('file_path XXX', customers, depotIndex, vehicleTours)
            #check(customerCount, customers, vehicleCapacity, vehicleTours)
            if not is_valid(customerCount, customers, vehicleCapacity, vehicleTours):
                continue
            dist = total_dist(customers, depotIndex, vehicleTours)
            if dist < dist0:
                print '%s => %s' % (dist0, dist)
                vehicleTours0 = vehicleTours[:]
                dist0 = dist
                
                
    vehicleTours = copy.deepcopy(vehicleTours0)  
    check(customerCount, customers, vehicleCapacity, vehicleTours)
    while len(vehicleTours) < vehicleCount:
        vehicleTours.append([])
        
    print '*', vehicleTours    
        
    return vehicleTours
    

def solveIt(inputData):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = inputData.split('\n')

    parts = lines[0].split()
    customerCount = int(parts[0])
    vehicleCount = int(parts[1])
    vehicleCapacity = int(parts[2])
    depotIndex = 0

    customers = []
    for i in range(1, customerCount+1):
        line = lines[i]
        parts = line.split()
        customers.append(Customer(int(parts[0]), float(parts[1]), float(parts[2])))


    vehicleTours = solve(customerCount, vehicleCount, vehicleCapacity, depotIndex, customers)
        
    assert len(vehicleTours) <= vehicleCount
    
    # checks that the number of customers served is correct
    assert sum([len(v) for v in vehicleTours]) == customerCount - 1

    # calculate the cost of the solution; for each vehicle the length of the route
    obj = 0
    for v in range(0, vehicleCount):
        vehicleTour = vehicleTours[v]
        if len(vehicleTour) > 0:
            obj += length(customers[depotIndex], customers[vehicleTour[0]])
            for i in range(0, len(vehicleTour) - 1):
                obj += length(customers[vehicleTour[i]], customers[vehicleTour[i + 1]])
            obj += length(customers[vehicleTour[-1]], customers[depotIndex])

    # prepare the solution in the specified output format
    outputData = str(obj) + ' ' + str(0) + '\n'
    for v in range(0, vehicleCount):
        outputData += str(depotIndex) + ' ' + ' '.join(map(str,vehicleTours[v])) + ' ' + str(depotIndex) + '\n'
        print '!', vehicleTours[v], sum(customers[i].demand for i in vehicleTours[v]), vehicleCapacity
    return outputData


problems = [    
 './data/vrp_16_3_1',
 './data/vrp_26_8_1',
 './data/vrp_51_5_1',
 './data/vrp_101_10_1',
 './data/vrp_200_16_1',
 './data/vrp_421_41_1']

print problems

def process_file(fileLocation):
   
    inputDataFile = open(fileLocation, 'r')
    inputData = ''.join(inputDataFile.readlines())
    inputDataFile.close()
    print 'Solving:', fileLocation
    print solveIt(inputData)
    
 
import sys

if __name__ == '__main__':

    if False:
        # problem vrp_26_8_1 is not fitting    
        for  fileLocation in problems[:1]:
            print fileLocation
            process_file(fileLocation)
            print fileLocation
            print '*' * 80
            
        exit()

    if len(sys.argv) > 1:
        fileLocation = sys.argv[1].strip()
        inputDataFile = open(fileLocation, 'r')
        inputData = ''.join(inputDataFile.readlines())
        inputDataFile.close()
        print 'Solving:', fileLocation
        print solveIt(inputData)
    else:

        print 'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)'

