#!/usr/bin/env python3

import numpy as np

def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    m, h, w, _ = A_prev.shape
    fh, fw, _, c = W.shape
    sh, sw = stride

    if padding == 'same':
        nh, nw = h, w
        ph  = int((fh + (sh * (h - 1)) - h) / 2 + 0.5)
        pw  = int((fw + (sw * (w - 1)) - w) / 2 + 0.5)
        A_prev = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')
    elif padding == 'valid':
        nh = ((h - fh) // sh) + 1
        nw = ((w - fw) // sw) + 1
    else:
        ph, pw = padding
        nh = ((h - fh + (2 * ph)) // sh) + 1
        nw = ((w - fw + (2 * pw)) // sw) + 1
        A_prev = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')
    Z = np.zeros((m, nh, nw, c))
    for i in range(nh):
        h_start = i * sh
        h_end = h_start + fh
        for j in range(nw):
            w_start = j * sw
            w_end = w_start + fw
            A_prev_slice = A_prev[:, h_start:h_end, w_start:w_end]
            for k in range(c):
                Z[:, i, j, k] = np.sum(A_prev_slice * W[:, :, :, k], axis=(1, 2, 3))
    A = activation(Z + b)
    return A
