#!/usr/bin/env python3

"""
Contains the function mean_cov
"""

import numpy as np

def mean_cov(X):
    """calculates the mean and covariance of X"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
            raise TypeError('X must be a 2D numpy.ndarray')
    n = X.shape[0]
    if n < 2:
        raise ValueError('X must contain multiple data points')
    mean = np.mean(X, axis=0, keepdims=True)
    cov = np.matmul((X - mean).T, (X - mean)) / (n - 1)
    return mean, cov
