import numpy as np

def sigmax():
    """
    Pauli-x operator
    """
    sigmax = np.array([[0, 1], 
                       [1, 0]])
    return sigmax

def sigmay():
    """
    Pauli-y operator
    """
    sigmay = np.array([[0, -1.0j], 
                       [1.0j, 0]])
    return sigmay

def sigmaz():
    sigmaz = np.array([[1, 0], 
                       [0, -1]])
    return sigmaz