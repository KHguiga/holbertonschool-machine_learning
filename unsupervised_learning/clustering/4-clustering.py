#!/usr/bin/env python3

import numpy as np
associated_clustering = __import__('1-clustering').associated_clustering
new_generation = __import__('2-clustering').new_generation
inertia = __import__('3-clustering').inertia


def K_means_with_given_gen_0(points , gen_0_centroids, n_iter=100):
    centroids = gen_0_centroids
    clustering = associated_clustering( points, centroids )
    inert = inertia( points, centroids, clustering )
    inertias = inert
    for i in range(n_iter):
        new_centroids = new_generation(points,centroids)
        clustering = associated_clustering( points, new_centroids )
        inert = inertia( points, new_centroids, clustering )
        if centroids.shape == new_centroids.shape and not (centroids - new_centroids).any():
            break
        inertias += inert
        centroids = new_centroids
    # return {'centroids': centroids, 'clustering': clustering, 'inertia': inert, 'iterations needed': i}
    return {'centroids': centroids, 'clustering': clustering, 'inertia': inert}