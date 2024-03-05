#!/usr/bin/env python3
"""Neural Style Transfer Module"""
import numpy as np
import tensorflow as tf


class NST:
    """Performs tasks for neural style transfer:

    Public class attributes:
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'
    Class constructor: def __init__(self, style_image, content_image,
    alpha=1e4, beta=1):
    style_image - the image used as a style reference, stored as a
    numpy.ndarray
    content_image - the image used as a content reference, stored as a
    numpy.ndarray
    alpha - the weight for content cost
    beta - the weight for style cost
    if style_image is not a np.ndarray with the shape (h, w, 3), raise a
    TypeError with the message style_image must be a numpy.ndarray with
    shape (h, w, 3)
    if content_image is not a np.ndarray with the shape (h, w, 3), raise
    a TypeError with the message content_image must be a numpy.ndarray with
    shape (h, w, 3)
    if alpha is not a non-negative number, raise a TypeError with the message
    alpha must be a non-negative number
    if beta is not a non-negative number, raise a TypeError with the message
    beta must be a non-negative number
    Sets the instance attributes:
    style_image - the preprocessed style image
    content_image - the preprocessed content image
    alpha - the weight for content cost
    beta - the weight for style cost
    """
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        if not isinstance(style_image, np.ndarray
                          ) or len(style_image.shape
                                   ) != 3 or style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(content_image, np.ndarray
                          ) or len(content_image.shape
                                   ) != 3 or content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(alpha, float) and not isinstance(alpha, int
                                                           ) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, float) and not isinstance(beta, int
                                                          ) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """Rescales an image such that its pixels values are between 0 and 1
        and its largest side is 512 pixels
        image - a numpy.ndarray of shape (h, w, 3) containing the image to be
        scaled
        if image is not a np.ndarray with the shape (h, w, 3), raise a
        TypeError with the message image must be a numpy.ndarray with
        shape (h, w, 3)
        The scaled image should be a tf.tensor with the shape
        (1, h_new, w_new, 3) where max(h_new, w_new) == 512 and
        min(h_new, w_new) is scaled proportionately
        The image should be resized using bicubic interpolation
        After resizing, the image’s pixel values should be rescaled from the
        range [0, 255] to [0, 1].
        Returns: the scaled
        """
        if not isinstance(image, np.ndarray
                          ) or len(image.shape) != 3 or image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w, _ = image.shape
        if h > w:
            h_new = 512
            w_new = w * h_new // h
        else:
            w_new = 512
            h_new = h * w_new // w

        scaled_image = tf.image.resize(image, tf.constant([h_new, w_new],
                                                          dtype=tf.int32),
                                       tf.image.ResizeMethod.BICUBIC)
        scaled_image = tf.reshape(scaled_image, (1, h_new, w_new, 3))
        scaled_image = tf.clip_by_value(scaled_image / 255, 0.0, 1.0)

        return scaled_image