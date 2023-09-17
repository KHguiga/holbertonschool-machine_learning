#!/usr/bin/env python3

import numpy as np

def squared_dists( A, B ):
    return np.sum(np.square(A[:,None,:]- B[None,:,:]), axis=2)
