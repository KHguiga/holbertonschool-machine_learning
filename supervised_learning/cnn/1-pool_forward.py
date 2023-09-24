#!/usr/bin/env python3

import numpy as np

def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    m, h, w, c = A_prev.shape
    fh, fw = kernel_shape
    sh, sw = stride
    nh = (h - fh) // sh + 1
    nw = (w - fw) // sw + 1
    A = np.zeros((m, nh, nw, c))
    for i in range(nh):
        h_start = i * sh
        h_end = h_start + fh
        for j in range(nw):
            w_start = j * sw
            w_end = w_start + fw
            A_prev_slice = A_prev[:, h_start:h_end, w_start:w_end, :]
            if mode == 'max':
                A[:, i, j, :] = np.max(A_prev_slice, axis=(1, 2))
            elif mode == 'avg':
                A[:, i, j, :] = np.mean(A_prev_slice, axis=(1, 2))
    return A
