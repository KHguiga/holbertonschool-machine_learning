#!/usr/bin/env python3

import numpy as np

def likelihood(x, n, P):
    """calculate the likelihood of our data"""
    if type(n) is not int or n < 1:
        raise ValueError('n must be a positive integer')
    if type(x) is not int or x < 0:
        raise ValueError('x must be an integer that is greater than or equal to 0')
    if x > n:
        raise ValueError('x cannot be greater than n')
    if type(P) is not np.ndarray or P.ndim != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    if np.min(P) < 0 or np.max(P) > 1:
        raise ValueError('All values in P must be in the range [0, 1]')
    y = n - x
    f = np.math.factorial(n) / (np.math.factorial(x) * np.math.factorial(y))
    return f * (P ** x) * ((1 - P) ** y)
