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
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

filename = "train.csv"
n = sum(1 for line in open(filename)) - 1 #number of records in file (excludes header)
s = 1000000 #desired sample size
skip = sorted(random.sample(xrange(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list
df = pd.read_csv(filename, skiprows=skip)
df.to_csv("train_sample.csv")



#os.remove("train_select3.csv")
#
#f=open("train.csv",'r')
#n = sum(1 for line in f) - 1 #number of records in file (excludes header)
#o=open("train_select3.csv", 'w')
#
#f.seek(0)
#random_line=f.readline()
#o.write(random_line)
#
#for i in range(0,1000000):
#    offset=random.randrange(n)
#    f.seek(offset)
#    f.readline()
#    random_line=f.readline()
#    o.write(random_line)
#
#qqqqqqq
#aaaa
#f.close()qweqweqweqwe
#o.close()
