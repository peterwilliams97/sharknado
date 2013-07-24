#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
  
          
"""
from __future__ import division
#from itertools import count
from collections import defaultdict
#import pprint
import re, os, time, shutil, glob


RE_PATH = re.compile('gc_(\d+)_(\d+)')
RESULTS_DIR = 'results'
SUMMARY_DIR = 'summary'
 
def base_name(path):    
    m = RE_PATH.search(path)
    if not m:
        return None
    return 'gc_%d_%d' % (int(m.group(1)), int(m.group(2)))

# 0: False: (41, -256, 4, -1623619902, (3,   ... ))
RE_RESULT = re.compile(r'(\d+):\s*(True|False):\s*\((-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),\s*\((.*?)\)\)')  
  
def parse_file(path):
    text = open(path, 'rt').read()
    results = []
    for m in RE_RESULT.finditer(text):
        #print m.groups()
        i, optimum, n, score, count, hsh, X = [eval(x) for x in m.groups()]
        #print  i, optimum, n, score, count, hsh, len(X)
        results.append((i, optimum, n, score, count, hsh, X))
    return results    

try:
    os.mkdir(SUMMARY_DIR)
except:
    pass    

files = list(glob.glob('results/*'))
print files

all_results = defaultdict(list)
for path in files:
    all_results[base_name(path)].extend(parse_file(path))
    
for k,v in all_results.items():
    print '-' * 80
    print k, len(v)
    v.sort(key=lambda x: (x[2], x[3]))
    for i, optimum, n, score, count, hsh, X in v[:3]:
        print i, n, score
    
for name, results in all_results.items():
    path = os.path.join(SUMMARY_DIR, name)
    with open(path, 'wt') as f:
        f.write('%s = [\n' % name) 
        for i, optimum, n, score, count, hsh, X in results:
            f.write('   %s,\n' % repr((n, score, X)))
        f.write(']\n\n')
   