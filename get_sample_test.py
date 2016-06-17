# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 20:57:41 2016

@author: can
"""


import random

import pandas as pd


filename = "input/test.csv"
n = sum(1 for line in open(filename)) - 1 #number of records in file (excludes header)
s = 10000 #desired sample size
skip = sorted(random.sample(xrange(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list
df = pd.read_csv(filename, skiprows=skip)
df.to_csv("input/test_sample_10000.csv")

