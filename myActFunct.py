import numpy as np

def identity(matrix, der=0):
    if der == 0:
        return matrix
    else:
        return matrix, 1
    
def tanh(matrix, der=0):
    res = np.tanh(matrix)
    if der == 0:
        return res
    else:
        return res, 1 - res*res 
    
def sigm(matrix, der=0):
    res = 1 / (1 + np.exp(-matrix))
    if der == 0:
        return res
    else:
        return res, res*(1-res)
    