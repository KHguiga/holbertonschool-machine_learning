#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

class NST:
    
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1', 
                    'block4_conv1', 
                    'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        tf.enable_eager_execution()
        if type(style_image) is not np.ndarray or style_image.ndim != 3 or style_image.shape[2] != 3:
            raise TypeError('style_image must be a numpy.ndarray with shape (h, w, 3)')
        if type(content_image) is not np.ndarray or content_image.ndim != 3 or content_image.shape[2] != 3:
            raise TypeError('content_image must be a numpy.ndarray with shape (h, w, 3)')
        if (type(alpha) is not int and type(alpha) is not float) or alpha < 0:
            raise TypeError('alpha must be a non-negative number')
        if (type(beta) is not int and type(beta) is not float) or beta < 0:
            raise TypeError('beta must be a non-negative number')
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        if type(image) is not np.ndarray or image.ndim != 3 or image.shape[2] != 3:
            raise TypeError('image must be a numpy.ndarray with shape (h, w, 3)')
        max_dims = 512
        shape = image.shape[:2]
        scale = max_dims / max(shape[0], shape[1])
        new_shape = (int(scale * shape[0]), int(scale * shape[1]))
        image = np.expand_dims(image, axis=0)
        image = tf.clip_by_value(tf.image.resize_bicubic(image, new_shape) / 255.0, 0.0, 1.0)
        return image