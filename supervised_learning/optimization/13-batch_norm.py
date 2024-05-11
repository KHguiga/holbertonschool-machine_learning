#!/usr/bin/env python3
"""
    update_variables_RMSProp
"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    m = np.mean(Z, axis=0)
    v = np.var(Z, axis=0)
    Z_norm = gamma * (Z - m) / np.sqrt(v + epsilon) + beta
    return Z_norm