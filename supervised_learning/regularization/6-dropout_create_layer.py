#!/usr/bin/env python3
"""Neural Style Transfer Module"""
import tensorflow.compat.v1 as tf

def dropout_create_layer(prev, n, activation, keep_prob):
    layer = tf.layers.Dense(n, activation=activation, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode=("fan_avg")), name='layer')
    A = layer(prev)
    drop = tf.layers.Dropout(1 - keep_prob)
    return drop(A)
