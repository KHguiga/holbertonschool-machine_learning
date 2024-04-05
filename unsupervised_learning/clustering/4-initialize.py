#!/usr/bin/env python3

import numpy as np
kmeans = __import__('1-kmeans').kmeans

def initialize(X, k):
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None, None
    if type(k) is not int or int(k) != k or k < 1:
        return None, None, None
    d = X.shape[1]
    pi = np.full(k, 1/k)
    m, _ = kmeans(X, k)
    S = np.tile(np.eye(d), (k, 1, 1))
    return pi, m, S
