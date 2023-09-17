#!/usr/bin/env python3

import numpy as np

def most_acute_angle(y):
    k = np.arange(1, len(y) - 1)
    angles = np.arccos(
        ((y[k + 1] - y[k]) * (y[k - 1] - y[k]) - 1) /
        np.sqrt(
            (((y[k + 1] - y[k])**2) + 1) * (((y[k] - y[k - 1])**2) + 1))
    )
    return np.argmin(angles)+1             
