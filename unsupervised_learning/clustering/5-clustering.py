#!/usr/bin/env python3

import numpy as np
K_means_with_given_gen_0 = __import__('4-clustering').K_means_with_given_gen_0

def random_init_centroids(points,K):
    
    n = points.shape[1]

    mi =np.min(points, axis=0)

    Ma = np.max(points, axis=0)

    P = np.random.rand(K*n).reshape((K,n))

    P *= (Ma - mi)[None, :]
    P += mi
    return P

def K_means( points , k, n_iter=100, n_init=100, seed=0):
    inert= 1.
    np.random.seed(seed)
    inertias=[]
    iterations_needed = 0
    for i in range(n_init):
        gen_0 = random_init_centroids(points,k)
        d = K_means_with_given_gen_0(points, gen_0, n_iter)
        inertias.append(d['inertia'])
        iterations_needed += d['iterations needed']
        if d['inertia'] < inert or i == 0 :
            inert = d['inertia']
            best_k = d
    inertias = np.array(inertias)
    best_k['mean inertia'] = np.mean(inertias)
    best_k['standard deviation inertias'] = np.std(inertias)
    best_k['mean iterations needed'] = np.mean(inertias)
    return best_k
