#!/usr/bin/env python3

import numpy as np
associated_clustering = __import__('1-clustering').associated_clustering

def new_generation(points, old_generation):
    clss = associated_clustering(points, old_generation)
    n = points.shape[0]
    k = old_generation.shape[0]
    centroids = []
    for i in range(k):
        cluster_points = points[clss == i]
        if cluster_points.size != 0:
            centroids.append(np.mean(cluster_points, axis=0))
    return np.asarray(centroids)