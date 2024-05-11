#!/usr/bin/env python3
"""
    learning_rate_decay
"""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    return alpha / (1 + decay_rate * (global_step // decay_step))