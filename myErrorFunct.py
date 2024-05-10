import numpy as np

def softMax(y):
    yExp = np.exp(y - y.max(0))
    res = yExp / sum(yExp) # sum(iterable, start)
    return res

def crossEntropy(y, true_y, der=0):
    res = softMax(y)
    if der == 0:
        return -(true_y * np.log(res)).sum()
    else:
        return res - true_y
    