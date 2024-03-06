#!/usr/bin/env python3
"""Neural Style Transfer Module"""
import numpy as np
import tensorflow as tf


class NST:
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
        self.model = self.load_model()
    @staticmethod
    def scale_image(image):
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

    def load_model(self):
        # Load our model. We load pretrained VGG, trained on imagenet data
        vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        
        # Convert MaxPooling2D to AveragePooling2D for style layers
        for layer in vgg.layers:
            if 'block' in layer.name and 'pool' in layer.name:
                layer.__class__ = tf.keras.layers.AveragePooling2D
        # Get output layers corresponding to style and content layers 
        model_outputs = [vgg.get_layer(name).output for name in self.style_layers]
        model_outputs.append(vgg.get_layer(self.content_layer).output)
        # Build model
        model = tf.keras.models.Model(vgg.input, model_outputs)
        self.model = model
        return model
    @staticmethod
    def gram_matrix(input_layer):
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or input_layer.shape.rank != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        _, h, w, c = input_layer.shape
        # Reshape the input_layer to (h*w, c)
        reshaped_layer = tf.reshape(input_layer, (-1, c))
        # Calculate the Gram matrix
        gram = tf.matmul(tf.transpose(reshaped_layer), reshaped_layer)
        # Normalize the Gram matrix
        gram /= tf.cast(h * w, tf.float32)
        # Add an extra dimension to match the required shape (1, c, c)
        gram = tf.expand_dims(gram, axis=0)

        return gram
