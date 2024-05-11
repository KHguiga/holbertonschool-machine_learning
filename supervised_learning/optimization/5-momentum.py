#!/usr/bin/env python3
"""
   momentum scratch
"""

def update_variables_momentum(alpha, beta1, var, grad, v):
    v = beta1 * v + (1 - beta1) * grad
    var -= alpha * v
    return var, v