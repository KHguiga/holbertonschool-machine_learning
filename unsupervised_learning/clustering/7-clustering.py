#!/usr/bin/env python3

import numpy as np
squared_dists = __import__('0-clustering').squared_dists

def SIV(points, centroids, clustering):
    m, n = points.shape
    k = centroids.shape[0]
    
    # Initialize SIV to 0
    siv = 0
    
    # Loop through each cluster
    for i in range(k):
        # Get points assigned to cluster i
        cluster_points = points[clustering == i]
        card_c = cluster_points.shape[0]
        # Calculate the squared Euclidean distance from each point to its centroid
        if card_c != 0:
            distances = squared_dists( cluster_points, np.array([centroids[i]]))
            cluster_variance = np.sum(distances)
            # Add the sum of squared distances to SIV
            siv += cluster_variance / card_c
    return siv