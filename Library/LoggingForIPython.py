def printIpython(data,head=5,level=1):
    '''Level = 1,2,3,4 -> 4 is print everything'''
    currentLevel = 2
    if(level <= currentLevel ):
        print data.head(head),'\n'