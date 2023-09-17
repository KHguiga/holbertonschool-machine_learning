#!/usr/bin/env python3

import numpy as np

def squared_dists(A, B):
    return np.sum(np.square(A[:, None, :] - B[None, :, :]), axis=2)

def gradient_of_density(points, sigma):
    
    def delta(q):
        m, n = points.shape
        r, _ = q.shape
        
        # Compute the squared Euclidean distances correctly
        squared_dist = squared_dists(points, q)
        
        # Compute the density values
        density = np.exp(-sigma * squared_dist)
        
        # Initialize an array to store the gradients
        gradients = np.zeros_like(q)
        
        for i in range(r):
            # Compute the gradient contributions for each point q[i]
            gradient_contributions = (2 * sigma / m) * density[:, i][:, np.newaxis] * (points - q[i])
            
            # Sum the gradient contributions along the points axis
            gradients[i] = np.sum(gradient_contributions, axis=0)
        
        return gradients
    
    return delta
