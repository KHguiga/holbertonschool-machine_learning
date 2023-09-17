#!/usr/bin/env python3

import numpy as np

def density(points, sigma):
    
    m, n = points.shape

    def delta(q):
        r, _ = q.shape
        density_values = np.zeros(r)

        for i in range(r):
            diff = points - q[i, :]  # Compute the difference between points and q[i, :]
            squared_distances = np.sum(diff**2, axis=1)  # Calculate squared distances
            density_values[i] = np.sum(np.exp(-sigma * squared_distances))

        return density_values / m

    return delta
