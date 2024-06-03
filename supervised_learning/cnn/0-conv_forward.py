#!/usr/bin/env python3
"""
project CNN
by Ced
"""
import numpy as np


def relu(Z):
    return np.maximum(Z, 0)

def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    to be continued
    """
    # A_prev is the input of the convolution layer
    m = A_prev.shape[0]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]
    # print("Al-1", m,h_prev,w_prev,c_prev)
    
    # W containing the kernels for the convolution
    kh = W.shape[0]
    kw = W.shape[1]
    c_prev = W.shape[2]
    c_new = W.shape[3]
    # print("Weights", kh, kw, c_prev, c_new)

    # define stride    
    sh = stride[0]
    sw = stride[1]

    if padding == 'valid':
        ph = 0
        pw = 0
    if padding == 'same':
        ph = int((sh * (h_prev - 1)  + kh - sh) / 2)
        pw = int((sw * (w_prev - 1)  + kw - sw) / 2)

    # out contains the output of convolution layer, careful may gives wrog out_h and out_w
    out_h = int((h_prev + 2 * ph - kh)/ sh + 1)
    out_w = int((w_prev + 2 * pw - kw)/ sw + 1)
    out_c = c_new
    conv = np.zeros((m,out_h,out_w,out_c))
    # print("out", out_h, out_w, out_c)

    # gives the padded input
    A_prev_pad = np.pad(A_prev,((0, 0), (ph, ph),(pw, pw),(0,0)))
    # print("a_prev_pad", A_prev_pad.shape)
    # print("Weights", W.shape)



    for i in range(out_h):
        for j in range(out_w):
            for c in range(out_c):
                # crop A_prev_pad by filter, the sum, 

                x = np.multiply(A_prev_pad[:,i*sh:i*sh+kh, j*sw:j*sw+kw,:], W[: ,: ,:,c])
                # print("x before sum",x.shape)
                x = np.sum(x, axis=(1,2,3)) 
                # print("x after",x.shape)
                
                conv[:, i, j, c] = x
    A = activation(conv+ b)         
    return A