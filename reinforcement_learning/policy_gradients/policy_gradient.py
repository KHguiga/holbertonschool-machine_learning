#!/usr/bin/env python3
"""
    Policy Gradient
"""

import numpy as np


def policy(state, weight):
    """
    """
    z = state.dot(weight)
    z -= np.max(z)
    exp = np.exp(z)

    return exp / np.sum(exp)


def softmax_grad(softmax):
    """
    """
    s = softmax.reshape(-1, 1)

    return np.diagflat(s) - np.dot(s, s.T)


def policy_gradient(state, weight):
    """
    """
    probs = policy(state, weight)
    action = np.random.choice(len(probs), p=probs)
    dsoftmax = softmax_grad(probs)[action, :]
    dlog = dsoftmax / probs[action]
    state = state.reshape(-1, 1)
    grad = state.dot(dlog[None, :])

    return action, grad
