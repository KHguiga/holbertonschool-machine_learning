#!/usr/bin/env python3

"""
Contains the function correlation
"""

import numpy as np


def correlation(C):
    """calculates the correlation matrix from the covariance matrix"""
    if type(C) is not np.ndarray:
        raise TypeError('C must be a numpy.ndarray')
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError('C must be a 2D square matrix')
    Co = np.zeros(C.shape)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if i == j:
                Co[i, j] = 1
            else:
                Co[i, j] = C[i, j] / np.sqrt(C[i, i] * C[j, j])
    return Co
