#!/usr/bin/env python3

import numpy as np

def squared_dists( A, B ):
    M = A.shape[0]
    N = B.shape[0]
    A_dots = (A*A).sum(axis=1).reshape((M,1))*np.ones(shape=(1,N))
    B_dots = (B*B).sum(axis=1)*np.ones(shape=(M,1))
    return  A_dots + B_dots -2*A.dot(B.T)

def associated_clustering( points, centroids ):
    return squared_dists(points, centroids).argmin(axis=1)

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