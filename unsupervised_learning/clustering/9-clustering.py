#!/usr/bin/env python3

import numpy as np
dist_based_clust = __import__('8-clustering').dist_based_clust

def dist_based_best_rho(points, rho_min, rho_max, n_sample=100):
    clusters = []
    max_count = 0
    count = 0
    start_idx = 0
    ks = np.arange(n_sample+1)
    rhos = rho_min + ks * (rho_max - rho_min) / n_sample
    for rho_k in rhos:
        clustering = dist_based_clust(points, rho_k)
        clusters.append(len(np.unique(clustering)))
    for i in range(1, len(clusters)):
        if clusters[i] == 1: # si nombre de clusters est 1 on sort de la boucle
            break
        if clusters[i] == clusters[i-1]:
            count += 1
            max_count = max(max_count, count)
        else:
            count = 0
            start_idx = i
    return (rhos[start_idx] + rhos[start_idx + max_count]) / 2
