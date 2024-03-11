#!/usr/bin/env python3
"""
    Neural style transfer
"""

import numpy as np
import tensorflow as tf


class NST:
    """
        Class that performs tasks for neural style transfer
    """

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
            Class constructor neural style transfer

            :param style_image: ndarray, image used as style reference
            :param content_image: ndarray, image used as content reference
            :param alpha: weight for content cost
            :param beta: weight for style cost
        """

        self.style_image = style_image

        if (not isinstance(style_image, np.ndarray)
                or style_image.shape[-1] != 3):
            raise TypeError("style_image must be a numpy.ndarray"
                            " with shape (h, w, 3)")
        else:
            self.style_image = self.scale_image(style_image)
        if (not isinstance(content_image, np.ndarray)
                or content_image.shape[-1] != 3):
            raise TypeError("content_image must be a numpy.ndarray"
                            " with shape (h, w, 3)")
        else:
            self.content_image = self.scale_image(content_image)
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        else:
            self.alpha = alpha
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")
        else:
            self.beta = beta

        self.model = None
        self.load_model()
        self.generate_features()

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
        vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
        x = vgg.input
        model_outputs = []
        content_output = None
        for layer in vgg.layers[1:]:
            if "pool" in layer.name:
                x = tf.keras.layers.AveragePooling2D(pool_size=layer.pool_size, strides=layer.strides, name=layer.name)(x)
            else:
                x = layer(x)
                if layer.name in self.style_layers:
                    model_outputs.append(x)
                if layer.name == self.content_layer:
                    content_output = x
                layer.trainable = False
        model_outputs.append(content_output)
        model = tf.keras.models.Model(vgg.input, model_outputs)
        self.model = model

    @staticmethod
    def gram_matrix(input_layer):
        """
            Calculate Gram Matrix

            :param input_layer: instance of tf.Tensor or tf.Variable
                shape(1,h,w,c), layer output whose gram matrix should
                be calculated
            :return: tf.tensor, shape(1,c,c) containing gram matrix
        """

        if (not isinstance(input_layer, (tf.Tensor, tf.Variable))
                or len(input_layer.shape) != 4):
            raise TypeError("input_layer must be a tensor of rank 4")

        # sum of product
        # b: num of batch, i&j spatial coordinate, c channel
        result = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)

        # form of input tensor
        input_shape = tf.shape(input_layer)

        # nbr spatial position in each feature card : h*w
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)

        # normalisation of result
        norm_result = result / num_locations

        return norm_result

    def generate_features(self):
        preprocessed_s = tf.keras.applications.vgg19.preprocess_input(self.style_image * 255)
        preprocessed_c = tf.keras.applications.vgg19.preprocess_input(self.content_image * 255)
        style_features = self.model(preprocessed_s)[:-1]
        self.content_feature = self.model(preprocessed_c)[-1]
        self.gram_style_features = [self.gram_matrix(style_feature) for style_feature in style_features]

    def layer_style_cost(self, style_output, gram_target):
        """
            method to calculate the style cost for a single layer

        :param style_output: tf.tensor, shape(1,h,w,c),
                    layer style output of the generated image
        :param gram_target: tf.tensor, shape(1,c,c)
                    gram matrix of the target style output for that layer

        :return: layer's style cost
        """

        if (not isinstance(style_output, (tf.Tensor, tf.Variable))
                or len(style_output.shape) != 4):
            raise TypeError("style_output must be a tensor of rank 4")

        _, _, _, c = style_output.shape

        if (not isinstance(gram_target, (tf.Tensor, tf.Variable))
                or gram_target.shape != [1, c, c]):
            raise TypeError(
                "gram_target must be a tensor of shape [1, {}, {}]".format(
                    c,
                    c
                ))

        output_gram_style = self.gram_matrix(style_output)

        # difference between two gram matrix
        layer_style_cost = tf.reduce_mean(
            tf.square(output_gram_style - gram_target))

        return layer_style_cost

    def style_cost(self, style_outputs):
        """
            methode to calculate style cost for generate image

            :param style_outputs: list of tf.tensor style outputs for
                generated image
            each layer should be weighted evenly with all weights summing to 1

            :return: style cost
        """
        len_style_layer = len(self.style_layers)
        if (not isinstance(style_outputs, list)
                or len(style_outputs) != len(self.style_layers)):
            raise TypeError(
                "style_outputs must be a list with a length of {}"
                .format(len_style_layer)
            )

        # uniform initialization
        weight = 1.0 / len_style_layer

        cost_total = sum([weight * self.layer_style_cost(style, target)
                         for style, target
                         in zip(style_outputs, self.gram_style_features)])

        return cost_total