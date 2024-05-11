#!/usr/bin/env python3
"""
   Batch Normalization upgraded
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    """
    kernel = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    layer = tf.keras.layers.Dense(n, activation=None,
                                  kernel_initializer=kernel)
    Z = layer(prev)

    beta = tf.Variable(initial_value=tf.zeros(shape=[n]), name="beta")
    gamma = tf.Variable(initial_value=tf.ones(shape=[n]), name="gamma")

    m, v = tf.nn.moments(Z, axes=[0])

    batch_norm = tf.nn.batch_normalization(Z, mean=m, variance=v, offset=beta,
                                           scale=gamma, variance_epsilon=1e-7)

    if activation is not None:
        batch_norm = activation(batch_norm)

    return batch_norm