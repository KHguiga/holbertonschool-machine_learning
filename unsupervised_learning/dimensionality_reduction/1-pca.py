#!/usr/bin/env python3

import numpy as np

def pca(X, ndim):
    """
    Takes X a matrix,
    return Tr( reduction score)
    """
    n = X.shape[0]  # number of data points
    d = X.shape[1]  # number of dimensions

    # normalize X
    X = X - np.mean(X, axis=0)

    # decompose X into SVD
    U, S, _ = np.linalg.svd(X, full_matrices=False)

    # reduction dimension r
    if ndim >= d:
        ndim = d
    Ur = U[:, 0:ndim]
    Sr = np.diag(S[0:ndim])
    T = Ur @ Sr
    return T
