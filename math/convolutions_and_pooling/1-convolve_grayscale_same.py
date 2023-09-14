#!/usr/bin/env python3

import numpy as np

def convolve_grayscale_same(images, kernel):
    m, h, w = images.shape
    fh, fw = kernel.shape

    ph  = int((fh - 1) / 2 + 0.5)
    pw  = int((fw - 1) / 2 + 0.5)
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')
    convolution = np.zeros((m, h, w))

    for i in range(h):
        h_start, h_end = i, i + fh
        for j in range(w):
            w_start, w_end = j, j + fw
            images_slice = images[:, h_start:h_end, w_start:w_end]
            convolution[:,i,j] = np.sum(images_slice * kernel, axis=(1, 2))
    return convolution
