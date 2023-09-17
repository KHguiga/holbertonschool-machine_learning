#!/usr/bin/env python3

import numpy as np
import networkx as nx

def calculate_distances(points):
    # Number of data points
    vectors = points[None,:,:]-points[:,None,:]
    return np.power(vectors,2).sum(axis=2)

def dist_based_clust(points, rho):
    # Calculate pairwise distances
    distances = calculate_distances(points)
    
    # Create a graph where edges connect points within the rho-neighborhood
    m = points.shape[0]
    G = nx.Graph()
    
    for i in range(m):
        for j in range(i, m):
            if distances[i,j] < rho**2:
                G.add_edge(i, j)
    
    # Find connected components in the graph
    components = list(nx.connected_components(G))
    s = len(components)
    clustering = [0]*m
    for cluster_idx, component in enumerate(components):
        for point_idx in component:
            clustering[point_idx] = cluster_idx
                
    # Create the clustering array
    return np.array(clustering)
