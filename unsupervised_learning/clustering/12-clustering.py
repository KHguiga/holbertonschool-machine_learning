#!/usr/bin/env python3

import numpy as np
gradient_of_density = __import__('11-clustering').gradient_of_density

def gradient_ascent_along_density(points, q, sigma, eta, n_steps):
    m, n = points.shape
    r, _ = q.shape
    
    # Initialize B with a copy of q
    B = np.copy(q)
    
    # Perform gradient ascent for all q points simultaneously for n_steps iterations
    for step in range(n_steps):
        # Compute the gradient of density for all q points
        gradients = gradient_of_density(points, sigma)(B)
        
        # Update all q points using gradient ascent
        B += eta * gradients
    
    return B