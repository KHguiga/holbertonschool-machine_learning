#!/usr/bin/env python3
"""
   Momentum upgraded
"""

import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    """
    optimizer = tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
    return optimizer