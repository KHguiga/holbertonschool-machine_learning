#!/usr/bin/env python3

import numpy as np

def squared_dists(A, B):
    M = A.shape[0]
    N = B.shape[0]
    A_dots = (A * A).sum(axis = 1).reshape((M, 1)) * np.ones(shape=(1, N))
    B_dots = (B * B).sum(axis = 1) * np.ones(shape = (M, 1))
    return  A_dots + B_dots - 2 * A.dot(B.T).astype(int)