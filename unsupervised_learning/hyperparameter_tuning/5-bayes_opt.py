#!/usr/bin/env python3
"""
    This module contains the implementation
    of the Bayesian Optimization Process
"""
import numpy as np
from scipy.stats import norm

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True):
        self.f = f
        self.gp = GaussianProcess(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        mu_s, sig_s = self.gp.predict(self.X_s)
        if self.minimize:
            fx_p = np.min(self.gp.Y)
            num = fx_p - mu_s - self.xsi
        else:
            fx_p = np.max(self.gp.Y)
            num = mu_s - fx_p - self.xsi

        Z = np.where(np.isclose(sig_s, 0), 0, num / sig_s)
        EI = np.where(np.isclose(sig_s, 0), 0, num * norm.cdf(Z) + sig_s * norm.pdf(Z))
        EI = np.maximum(EI, 0)
        X_next = self.X_s[np.argmax(EI)][0].reshape(-1)
        return X_next, EI

    def optimize(self, iterations=100):
        for _ in range(iterations):
            X_next, _ = self.acquisition()
            if X_next in self.gp.X:
                break
            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)
        if self.minimize:
            i_opt = np.argmin(self.gp.Y)
        else:
            i_opt = np.argmax(self.gp.Y)
        return self.gp.X[i_opt], self.gp.Y[i_opt]