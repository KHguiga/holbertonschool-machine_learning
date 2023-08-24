#!/usr/bin/env python3
"""
module containing function calculate_loss
"""
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """
    calculates the softmax loss of a prediction
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)
