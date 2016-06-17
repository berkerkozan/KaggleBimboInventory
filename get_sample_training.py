# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#import pandas
#import random
#
#filename = "train.csv"
#n = sum(1 for line in open(filename)) - 1 #number of records in file (excludes header)
#s = 10000 #desired sample size
#skip = sorted(random.sample(xrange(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list
#df = pandas.read_csv(filename, skiprows=skip)


import random

import pandas as pd


filename = "input/train.csv"
n = sum(1 for line in open(filename)) - 1 #number of records in file (excludes header)
s = 1000000 #desired sample size
skip = sorted(random.sample(xrange(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list
df = pd.read_csv(filename, skiprows=skip)
df.to_csv("input/train_sample_1000000.csv")

