#!/usr/bin/env python3
"""
   RMSProp upgraded
"""

import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    """
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=alpha,
                                            rho=beta2,
                                            epsilon=epsilon)
    return optimizer