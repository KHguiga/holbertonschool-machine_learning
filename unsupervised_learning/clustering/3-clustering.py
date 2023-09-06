#!/usr/bin/env python3

import numpy as np
kmeans = __import__('0-clustering').squared_dists


def inertia(points, centroids, clustering):
    k = centroids.shape[0]
    inertia = 0
    for i in range(k):
        cluster_points = points[clustering == i]
        cluster_variance = np.sum(squared_dists(cluster_points, np.array([centroids[i]]))) 
        inertia += cluster_variance
    return inertia / points.shape[0]