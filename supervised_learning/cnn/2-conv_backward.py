#!/usr/bin/env python3

import numpy as np

def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    m, h_prev, w_prev, _ = A_prev.shape
    _, h_new, w_new, c_new = dZ.shape
    fh, fw, _, _ = W.shape
    sh, sw = stride
    if padding == "same":
        ph  = int((fh + (sh * (h_prev - 1)) - h_prev) / 2 + 0.5)
        pw  = int((fw + (sw * (w_prev - 1)) - w_prev) / 2 + 0.5)
    elif padding == "valid":
        ph, pw = 0, 0
    else:
        ph, pw = padding
    A_prev_pad = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)
    dA_prev_pad = np.zeros(A_prev_pad.shape)

    for i in range(m):
        for h in range(h_new):
            h_start = h * sh
            h_end = h_start + fh
            for w in range(w_new):
                w_start = w * sw
                w_end = w_start + fw
                for c in range(c_new):
                    dA_prev_pad[i, h_start:h_end, w_start:w_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += A_prev_pad[i, h_start:h_end, w_start:w_end, :] * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

    if A_prev_pad.shape == A_prev.shape:
        return dA_prev_pad, dW, db
    if ph == 0:
        return dA_prev_pad[:, :, pw:-pw, :], dW, db
    if pw == 0:
        return dA_prev_pad[:, ph:-ph, :, :], dW, db
    return dA_prev_pad[:, ph:-ph, pw:-pw, :], dW, db
