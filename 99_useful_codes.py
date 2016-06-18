

# Custom Scoring for Bimbo (for pred vs truth)

def rmsle_func(truths, preds):
    truths = np.asarray(truths)
    preds = np.asarray(preds)
    n = len(truths)
    diff = (np.log(preds+1) - np.log(truths+1))**2
    print(diff, n, np.sum(diff))
    return np.sqrt(np.sum(diff)/n)