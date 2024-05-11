#!/usr/bin/env python3
"""
   Learning Rate decay upgraded
"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    """
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_rate=decay_rate,
        decay_steps=decay_step,
        staircase=True)

    return lr_schedule