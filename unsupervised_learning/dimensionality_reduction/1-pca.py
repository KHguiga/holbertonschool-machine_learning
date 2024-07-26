#!/usr/bin/env python3
"""
PCA v2
"""

import numpy as np


def pca(X, ndim):
    # Center data (X) by subtracting the mean of each feature
    data_centered = X - np.mean(X, axis=0)

    # Compute the SVD of the data matrix
    U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)

    # Transposed (for shape(d, nd)) top ndim rows of Vt
    W = Vt[:ndim].T

    # Return transformed data in new dimensions/space
    return data_centered @ W