#!/usr/bin/env python3
"""
   RMSProp upgraded
"""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    op = tf.train.RMSPropOptimizer(alpha, beta2, epsilon=epsilon)
    return op.minimize(loss)