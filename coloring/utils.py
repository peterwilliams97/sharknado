#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    113  3967128: 0, 3, 4, 8, 9, 25, 109, value=3967128, capacity=5

"""
from __future__ import division
import sys, os, random
import numpy as np
import math

    
from itertools import count
from collections import deque
import bisect

class SortedDeque(deque):

    def __init__(self, iterable, maxlen):
        super(SortedDeque, self).__init__(sorted(iterable), maxlen)

    def _insert(self, index, value):
        
        #print ' 1>', self, index, value 
        self.rotate(-index)
        #print ' 2>', self, index, value
        self.appendleft(value)
        #print ' 3>', self, index, value
        self.rotate(index)
        #print ' 4>', self, index, value
        #assert all(self[i-1] <= self[i] for i in range(1, len(self)))

    def insert(self, value):
        if len(self) >= self.maxlen:
            if value > self[-1]:
                return
            self.pop()
        self._insert(bisect.bisect_left(self, value), value)

if False:        
    d = SortedDeque([1,5,3], 3)
    print d
    for i in range(7):
        d.insert(i)
        print i, d, d[-1]
    exit()
    
class LRU:

    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.l_set = set([])
        self.l_deque = deque()
        
    def add(self):
        n = len
        
