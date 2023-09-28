#!/usr/bin/env python3
import tensorflow.keras as K

def dense_block(X, nb_filters, growth_rate, layers):

    H = X

    for _ in range(layers):
        # 1x1 conv
        X = K.layers.BatchNormalization(axis=3)(H)
        X = K.layers.Activation('relu')(X)
        X = K.layers.Conv2D(growth_rate * 4, (1, 1), padding='same', kernel_initializer=K.initializers.he_normal())(X)

        # 3x3 conv
        X = K.layers.BatchNormalization(axis=3)(X)
        X = K.layers.Activation('relu')(X)
        X = K.layers.Conv2D(growth_rate, (3, 3), padding='same', kernel_initializer=K.initializers.he_normal())(X)

        # concatenate all outputs of dense block
        H = K.layers.Concatenate(axis=3)([H, X])
        nb_filters += growth_rate

    return H, nb_filters