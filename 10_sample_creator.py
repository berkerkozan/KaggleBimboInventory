import random
import timeit

sampleSizeList = [100,1000,1000000] # Sample Size
# test = '../input/test_alld_totals.csv'
# train = '../input/train_alld_totals.csv'

test = '../input/test.csv'
train = '../input/train.csv'

from Library import FileOperations

random.seed(1)

for sampleSize in sampleSizeList:
    print(timeit.timeit(lambda:FileOperations.createSampleFile(test,sampleSize),number=1))
    print(timeit.timeit(lambda:FileOperations.createSampleFile(train,sampleSize),number=1))