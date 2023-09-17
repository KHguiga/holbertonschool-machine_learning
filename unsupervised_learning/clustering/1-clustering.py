#!/usr/bin/env python3

import numpy as np
squared_dists = __import__('0-clustering').squared_dists

def associated_clustering( points, centroids ):
    return squared_dists(points, centroids).argmin(axis=1)