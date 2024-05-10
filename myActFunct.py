import numpy as np

# Matrix -> input
def identity(matrix, der=0):
    if der == 0:
        return matrix
    else:
        return matrix, 1
    
def tanh(matrix, der=0):
    x = np.exp(-2 * matrix)
    res = (1 - x)/(1 + x)
    if der == 0:
        return res
    else:
        return res, 1 - res*res # Derivata tangente iperbolica
    
def sigm(matrix, der=0):
    res = 1 / (1 + np.exp(-matrix))
    if der == 0:
        return res
    else:
        return res, res*(1-res)
    