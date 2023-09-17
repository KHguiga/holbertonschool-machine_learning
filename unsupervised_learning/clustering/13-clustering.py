#!/usr/bin/env python3

import numpy as np
dist_based_clust = __import__('8-clustering').dist_based_clust
gradient_ascent_along_density = __import__('12-clustering').gradient_ascent_along_density


def density_based_clustering( points, sigma, n_steps=200, eta_descent=2.0, rho_dist_based=.1):
    points = points.astype(float)
    q_end = gradient_ascent_along_density(points, points, sigma, eta_descent, n_steps)
    return dist_based_clust(q_end, rho_dist_based)
