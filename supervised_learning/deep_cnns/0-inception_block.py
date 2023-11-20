#!/usr/bin/env python3

import tensorflow.keras as K

def inception_block(A_prev, filters):
    """Performs K-means on a dataset:

    X is a numpy.ndarray of shape (n, d) containing the dataset
    k is the number of clusters
    The only import you are allowed to use is import sklearn.cluster
    Returns: C, clss
    C is a numpy.ndarray of shape (k, d) containing the centroid means for
    each cluster
    clss is a numpy.ndarray of shape (n,) containing the index of the cluster
    in C that each data point belongs to
    """

    F1, F3R, F3, F5R, F5, FPP = filters

    B1 = K.layers.Conv2D(F1, (1, 1), padding='same', activation='relu')(A_prev)

    B2 = K.layers.Conv2D(F3R, (1, 1), padding='same', activation='relu')(A_prev)
    B2 = K.layers.Conv2D(F3, (3, 3), padding='same', activation='relu')(B2)

    B3 = K.layers.Conv2D(F5R, (1, 1), padding='same', activation='relu')(A_prev)
    B3 = K.layers.Conv2D(F5, (5, 5), padding='same', activation='relu')(B3)

    B4 = K.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(A_prev)
    B4 = K.layers.Conv2D(FPP, (1, 1), padding='same', activation='relu')(B4)

    return K.layers.Concatenate()([B1, B2, B3, B4])