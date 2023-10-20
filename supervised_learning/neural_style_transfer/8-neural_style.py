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
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

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
        base_model = tf.keras.applications.VGG19(include_top=False)
        base_model.trainable = False

        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        base_model.save('base_model')
        base_model = tf.keras.models.load_model('base_model',
                                                custom_objects=custom_objects)

        outputs = [base_model.get_layer(layer
                                        ).output for layer in self.style_layers]
        outputs.append(base_model.get_layer(self.content_layer).output)
        model = tf.keras.Model(base_model.inputs, outputs)

        self.model = model

    @staticmethod
    def gram_matrix(input_layer):
        if not isinstance(input_layer, (tf.Variable, tf.Tensor)) or len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        result = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)

        input_shape = tf.shape(input_layer)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)

        return result / num_locations

    def generate_features(self):
        style_inputs = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255)
        content_inputs = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255)

        style_outputs = self.model(style_inputs)
        content_outputs = self.model(content_inputs)

        self.gram_style_features = [self.gram_matrix(
            style_layer) for style_layer in style_outputs[:-1]]
        self.content_feature = content_outputs[-1]

    def layer_style_cost(self, style_output, gram_target):
        if not isinstance(style_output, (tf.Variable, tf.Tensor)) or len(style_output.shape) != 4:
            raise TypeError("style_output must be a tensor of rank 4")

        _, _, _, c = style_output.shape

        if not isinstance(gram_target, (tf.Variable, tf.Tensor)) or gram_target.shape != (1, c, c):
            raise TypeError(
                f"gram_target must be a tensor of shape [1, {c}, {c}]")

        gram_style = self.gram_matrix(style_output)

        style_loss_layer = tf.math.reduce_mean(
            tf.square(
                tf.subtract(gram_target, gram_style)))

        return style_loss_layer

    def style_cost(self, style_outputs):
        style_len = len(self.style_layers)
        if not isinstance(style_outputs, list
                          ) or len(style_outputs) != style_len:
            raise TypeError(
                f"style_outputs must be a list with a length of {style_len}")

        style_outputs_cost = tf.add_n([self.layer_style_cost(
            style_output, self.gram_style_features[i]
        ) for i, style_output in enumerate(style_outputs)])

        return style_outputs_cost * (self.beta / style_len)

    def content_cost(self, content_output):
        content_shape = self.content_feature.shape

        if not isinstance(content_output, (tf.Variable, tf.Tensor)) or content_output.shape != content_shape:
            raise TypeError(
                f"content_output must be a tensor of shape {content_shape}")

        content_cost = tf.math.reduce_mean(
            tf.square(
                tf.subtract(self.content_feature, content_output)))

        return content_cost

    def total_cost(self, generated_image):
        content_shape = self.content_image.shape
        if not isinstance(generated_image, (tf.Variable, tf.Tensor)) or generated_image.shape != content_shape:
            raise TypeError(
                f"generated_image must be a tensor of shape {content_shape}")

        preprocessed = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255)
        outputs = self.model(preprocessed)
        content_output = outputs[-1]
        style_outputs = outputs[:-1]

        content_cost = self.content_cost(content_output)
        style_cost = self.style_cost(style_outputs)
        total_cost = self.alpha * content_cost + self.beta * style_cost

        return (total_cost, content_cost, style_cost)

    def compute_grads(self, generated_image):
        content_shape = self.content_image.shape
        if not isinstance(generated_image, (tf.Variable, tf.Tensor)) or generated_image.shape != content_shape:
            raise TypeError(
                f"generated_image must be a tensor of shape {content_shape}")

        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            J_total, J_content, J_style = self.total_cost(
                generated_image)
        grad = tape.gradient(J_total, generated_image)

        return (grad, J_total, J_content, J_style)