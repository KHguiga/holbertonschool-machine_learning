#!/usr/bin/env python3

import numpy as np

def convolve(images, kernels, padding='same', stride=(1, 1)):
    m, h, w, _ = images.shape
    fh, fw, _, c = kernels.shape
    sh, sw = stride

    if padding == 'same':
        nh, nw = h, w
        ph  = int((fh + (sh * (h - 1)) - h) / 2 + 0.5)
        pw  = int((fw + (sw * (w - 1)) - w) / 2 + 0.5)
        images = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')
    elif padding == 'valid':
        nh = ((h - fh) // sh) + 1
        nw = ((w - fw) // sw) + 1
    else:
        ph, pw = padding
        nh = ((h - fh + (2 * ph)) // sh) + 1
        nw = ((w - fw + (2 * pw)) // sw) + 1
        images = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')
    convolution = np.zeros((m, nh, nw, c))
    for i in range(nh):
        h_start = i * sh
        h_end = h_start + fh
        for j in range(nw):
            w_start = j * sw
            w_end = w_start + fw
            images_slice = images[:, h_start:h_end, w_start:w_end]
            for k in range(c):
                convolution[:, i, j, k] = np.sum(images_slice * kernels[:, :, :, k], axis=(1, 2, 3))
    return convolution
