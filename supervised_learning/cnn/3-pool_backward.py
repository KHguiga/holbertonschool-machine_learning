#!/usr/bin/env python3

import numpy as np

def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    m, h_new, w_new, c_new = dA.shape
    fh, fw = kernel_shape
    sh, sw = stride
    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        for h in range(h_new):
            h_start = h * sh
            h_end = h_start + fh
            for w in range(w_new):
                w_start = w * sw
                w_end = w_start + fw
                for c in range(c_new):
                    A_prev_slice = A_prev[i, h_start:h_end, w_start:w_end, c]
                    if mode == "max":
                        mask = (A_prev_slice == np.max(A_prev_slice))
                        dA_prev[i, h_start:h_end, w_start:w_end, c] += mask * dA[i, h, w, c]
                    elif mode == "avg":
                        dA_prev[i, h_start:h_end, w_start:w_end, c] += dA[i, h, w, c] / (fh * fw)
    return dA_prev
