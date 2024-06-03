#!/usr/bin/env python3
"""This module preforms forwarsd propagation over a
convutional layer of a neural network"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    m, h_prev, w_prev, c_prev = A_prev.shape

    # Extract dimensions from W's shape
    kh, kw, _, c_new = W.shape

    # Extract strides
    sh, sw = stride

    # Calculate padding for 'same' and 'valid'
    if padding == 'same':
        ph = int(((h_prev - 1) * sh - h_prev + kh) / 2 + 0.5)
        pw = int(((w_prev - 1) * sw - w_prev + kw) / 2 + 0.5)
    elif padding == 'valid':
        ph, pw = 0, 0

    # Initialize output with zeros
    output = np.zeros((m, int((h_prev - kh + 2 * ph) / sh) + 1,
                       int((w_prev - kw + 2 * pw) / sw) + 1, c_new))

    # Pad A_prev
    A_prev = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')

    # Convolve the input with the kernel and apply activation function
    for i in range(output.shape[1]):
        for j in range(output.shape[2]):
            for k in range(c_new):
                slice_A_prev = \
                    A_prev[:, i * sh: i * sh + kh, j * sw: j * sw + kw, :]
                conv = (W[..., k] * slice_A_prev).sum(axis=(1, 2, 3))
                output[:, i, j, k] = activation(conv + b[..., k])

    return output