#!/usr/bin/env python3
"""Connvolutional backward prop"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """ Performs backward propagation over a convolutional layer of a neural network

    Parameters:
    dZ (numpy.ndarray): Gradient of the cost with respect to the output of the
                        convolutional layer. Shape (m, h, w, c) where m is the
                        number of examples, h and w are the height and width of
                        the output, and c is the number of channels.
    A_prev (numpy.ndarray): Output from the previous layer. Shape (m, h_prev,
                            w_prev, c_prev) where m is the number of examples,
                            h_prev and w_prev are the height and width of the
                            previous layer, and c_prev is the number of channel
    W (numpy.ndarray): Weights for the current layer. Shape (kh, kw, kc, knc)
                       where kh and kw are the height and width of the filter,
                       kc is the number of channels in the previous layer, and
                       knc is the number of filters.
    b (numpy.ndarray): Biases for the current layer.
    padding (str): Type of padding to be used, either 'same' or 'valid'.
    stride (tuple): Tuple of (sh, sw) representing stride height and width.

    Returns:
    dA_prev (numpy.ndarray): Gradient of the cost with respect to the output of
                             the previous layer (A_prev). Shape (m, h_prev, w_prev, c_prev)
    dW (numpy.ndarray): Gradient of the cost with respect to the weights of the
                        current layer (W). Shape (kh, kw, kc, knc)
    db (numpy.ndarray): Gradient of the cost with respect to the biases of the
                       current layer (b). Shape (1, 1, 1, knc)
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2 + 0.5)
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2 + 0.5)
    else:  # padding == 'valid':
        ph = pw = 0

    A_prev_pad = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')
    dA_prev_pad = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(h_new):
        for j in range(w_new):
            for k in range(c_new):
                slice_A_prev = A_prev_pad[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
                dA_prev_pad[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :] += W[..., k] * dZ[:, i, j, k, np.newaxis, np.newaxis, np.newaxis]
                dW[..., k] += np.sum(slice_A_prev * dZ[:, i, j, k, np.newaxis, np.newaxis, np.newaxis], axis=0)

    if padding == 'same':
        dA_prev = dA_prev_pad[:, ph:-ph, pw:-pw, :]
    else:  # padding == 'valid':
        dA_prev = dA_prev_pad

    return dA_prev, dW, db