#!/usr/bin/env python3

import tensorflow as tf

def l2_reg_cost(cost):
    return cost + tf.losses.get_regularization_losses()
