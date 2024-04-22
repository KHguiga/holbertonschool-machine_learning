#!/usr/bin/env python3

"""
Contains the function mean_cov
"""

import numpy as np

class GaussianProcess:
    def __init__(self, X, Y, l=1, sigma_f=1):
        self.X = X
        self.Y = Y
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X, X)

    def kernel(self, X1, X2):
        K = (self.sigma_f**2) * np.exp(np.square(X1 - X2.T) / -(2 * (self.l ** 2)))
        return K