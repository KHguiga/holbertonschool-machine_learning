#!/usr/bin/env python3
"""This module contains the function for updating a variable
using the Adam
optimization algorithm
"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Update the learning rate using inverse time decay in numpy.

    Parameters:
    alpha (float): The original learning rate.
    decay_rate (float): The weight used to determine the rate at
    which alpha will decay.
    global_step (int): The number of passes of gradient descent that
    have elapsed.
    decay_step (int): The number of passes of gradient descent that should
    ccur before alpha is decayed further.

    Returns:
    float: The updated value for alpha.
    """

    # Calculate the decay factor
    decay_factor = 1 / (1 + decay_rate * (global_step // decay_step))

    # Update the learning rate
    alpha_updated = alpha * decay_factor

    # Return the updated learning rate
    return alpha_updated