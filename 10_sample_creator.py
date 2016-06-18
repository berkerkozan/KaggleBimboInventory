import random
import pandas as pd


s = 10000 # Sample Size

test = '../input/test.csv'
train = '../input/train.csv'

random.seed(1)

n = sum(1 for line in open(test)) - 1 #number of records in file (excludes header)
skip = sorted(random.sample(xrange(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list
df = pd.read_csv(test, skiprows=skip)
df.to_csv('../input/test_sample_10000.csv')

# todo dosya adını sample size'den otomaitk oluşturmak güzel olur.


n = sum(1 for line in open(train)) - 1 #number of records in file (excludes header)
skip = sorted(random.sample(xrange(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list
df = pd.read_csv(train, skiprows=skip)
df.to_csv('../input/train_sample_10000.csv')