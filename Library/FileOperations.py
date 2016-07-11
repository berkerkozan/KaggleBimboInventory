import random
import pandas as pd

def createSampleFile(filePath,sampleSize):
    '''trainOrTest = write test or train to work..'''
    with open(filePath) as f:
        totalLine = sum(1 for _ in f)
    print "There are {} lines..".format(totalLine)

    skip = (random.sample(xrange(1,totalLine+1),totalLine-sampleSize)) #the 0-indexed header will not be included in the skip list
    df = pd.read_csv(filePath, skiprows=skip)

    filePath = filePath.replace("train", "train_" + str(sampleSize)).replace("test", "test_" + str(sampleSize))
    df.to_csv(filePath,index=False)
    print "created "+ filePath
