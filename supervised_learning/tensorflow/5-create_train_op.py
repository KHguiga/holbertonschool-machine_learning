#!/usr/bin/env python3
"""
containes method create_train_op
"""
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha,
                                                  name='GradientDescent')
    train = optimizer.minimize(loss)
    return train
