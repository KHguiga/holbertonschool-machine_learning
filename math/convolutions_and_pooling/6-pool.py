#!/usr/bin/env python3

import numpy as np

def pool(images, kernel_shape, stride, mode='max'):
    m, h, w, c = images.shape
    fh, fw = kernel_shape
    sh, sw = stride
    nh = (h - fh) // sh + 1
    nw = (w - fw) // sw + 1
    pooling = np.zeros((m, nh, nw, c))

    for i in range(nh):
        h_start = i * sh
        h_end = h_start + fh
        for j in range(nw):
            w_start = j * sw
            w_end = w_start + fw
            images_slice = images[:, h_start:h_end, w_start:w_end, :]
            if mode == 'max':
                pooling[:, i, j, :] = np.max(images_slice, axis=(1, 2))
            elif mode == 'avg':
                pooling[:, i, j, :] = np.mean(images_slice, axis=(1, 2))
    return pooling
