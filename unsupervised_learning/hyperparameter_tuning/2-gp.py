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

    def predict(self, X_s):
        K_s = self.kernel(self.X, X_s)
        K_inv = np.linalg.inv(self.K)
        mu_s = np.matmul(np.matmul(K_s.T, K_inv), self.Y).reshape(-1)
        sig_s = self.sigma_f**2 - np.sum(np.matmul(K_s.T, K_inv).T * K_s, axis=0)
        return mu_s, sig_s

    def update(self, X_new, Y_new):
        self.X = np.concatenate([self.X, X_new.reshape(1, 1)], axis=0)
        self.Y = np.concatenate([self.Y, Y_new.reshape(1, 1)], axis=0)
        self.K = self.kernel(self.X, self.X)
