#!/usr/bin/env python3
"""Neural Style Transfer project by Mateo Quirós Asprón"""
import numpy as np
import tensorflow as tf


class NST:
    """Neural style transfer class"""
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initializes the class"""
        if not isinstance(style_image, np.ndarray):
            raise TypeError("style_image must be a numpy.ndarray \
with shape (h, w, 3)")

        if style_image.shape[-1] != 3:
            raise TypeError("style_image must be a numpy.ndarray \
with shape (h, w, 3)")

        if not isinstance(content_image, np.ndarray):
            raise TypeError("content_image must be a numpy.ndarray \
with shape (h, w, 3)")

        if content_image.shape[-1] != 3:
            raise TypeError("content_image must be a numpy.ndarray \
with shape (h, w, 3)")

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
        """Rescales the image"""
        if not isinstance(image, np.ndarray) or image.shape[-1] != 3:
            raise TypeError("image must be a numpy.ndarray \
with shape (h, w, 3)")
        height, width = image.shape[:2]
        max_side = np.maximum(height, width)
        scaling_factor = 512/max_side
        new_height = int(height * scaling_factor)
        new_width = int(width * scaling_factor)
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.expand_dims(image, axis=0)
        rescaled_image = tf.image.resize(image, (new_height,
                                         new_width), method='bicubic')
        rescaled_image = rescaled_image / 255.0
        rescaled_image = tf.clip_by_value(rescaled_image, 0, 1)
        return rescaled_image

    def load_model(self):
        """Loads the model that will be used for feature extraction"""
        vgg = tf.keras.applications.vgg19.VGG19(include_top=False,
                                                weights='imagenet')
        vgg.trainable = False

        # We save the model and replace max pooling with average pooling
        vgg.save("VGG19_base_model")
        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        vgg = tf.keras.models.load_model("VGG19_base_model",
                                         custom_objects=custom_objects)

        # Gets the desired style layers from the model
        style_outputs = [vgg.get_layer(name).output for
                         name in self.style_layers]
        # Gets the desired content layer from the model
        content_output = vgg.get_layer(self.content_layer).output
        # Unites the layers for our architecture
        model_outputs = style_outputs + [content_output]
        # Builds the model with our desired layers
        model = tf.keras.models.Model(inputs=vgg.input,
                                      outputs=model_outputs)

        self.model = model

    @staticmethod
    def gram_matrix(input_layer):
        """Calculates gram matrices"""
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)):
            raise TypeError("input_layer must be a tensor of rank 4")
        if tf.rank(input_layer) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")
        _, h, w, c = input_layer.shape
        # Flatten spatial dimensions
        flattened_layer = tf.reshape(input_layer, [1, h*w, c])
        # Compute Gram matrix
        gram = tf.matmul(flattened_layer, flattened_layer, transpose_a=True)
        # Normalize by number of spatial locations
        gram = gram / tf.cast(h*w, tf.float32)
        return gram

    def generate_features(self):
        """extracts the features used to calculate neural style cost"""
        # Load the VGG19 model for preprocessing
        VGG19 = tf.keras.applications.vgg19
        # Preprocessing fucntion expect the values in range [0, 255]
        style_image = VGG19.preprocess_input(self.style_image * 255)
        # Our architecutre only uses the last layer for content
        style_features = self.model(style_image)[:-1]
        style_matrix = [self.gram_matrix(tf.convert_to_tensor(feature))
                        for feature in style_features]

        content_image = VGG19.preprocess_input(self.content_image * 255)
        content_feature = self.model(content_image)[-1]

        self.gram_style_features = style_matrix
        self.content_feature = content_feature

    def layer_style_cost(self, style_output, gram_target):
        "Calculates the style cost for a single layer"""
        if not isinstance(style_output, (tf.Tensor, tf.Variable)):
            raise TypeError("style_output must be a tensor of rank 4")

        if tf.rank(style_output) != 4:
            raise TypeError("style_output must be a tensor of rank 4")

        # Number of channels
        c = style_output.shape[-1]

        if not isinstance(gram_target, (tf.Tensor, tf.Variable)):
            raise TypeError(f"gram_target must be a \
tensor of shape [1, {c}, {c}]")

        if gram_target.shape != (1, c, c):
            raise TypeError(f"gram_target must be a \
tensor of shape [1, {c}, {c}]")

        gram_output = self.gram_matrix(style_output)
        style_cost = tf.reduce_mean(tf.square(gram_output - gram_target))
        return style_cost

    def style_cost(self, style_outputs):
        """Calculates the total style cost"""
        length = len(style_outputs)
        len_style = len(self.style_layers)

        if not isinstance(style_outputs, list) or length != len_style:
            raise TypeError(f"style_outputs must be \
a list with a length of {len_style}")

        # Sets the weight so that they sum to one
        weight = 1 / length
        style_cost = 0
        # We get the cost of each layer and add it to the total
        for i in range(length):
            style_cost += weight * self.layer_style_cost(
                style_outputs[i], self.gram_style_features[i])
        return style_cost

    def content_cost(self, content_output):
        """Calculates the content cost"""
        feature_shape = self.content_feature.shape

        if not isinstance(content_output, (tf.Tensor, tf.Variable)):
            raise TypeError(f"content_output must be a tensor \
of shape {feature_shape}")

        if content_output.shape != feature_shape:
            raise TypeError(f"content_output must be a tensor \
of shape {feature_shape}")

        content_target = self.content_feature
        content_cost = tf.reduce_mean(
                       tf.square(content_output - content_target))
        return content_cost

    def total_cost(self, generated_image):
        """Calculates the total cost"""
        content_shape = self.content_image.shape

        if not isinstance(generated_image, (tf.Tensor, tf.Variable)):
            raise TypeError(f"generated_image must be a tensor \
of shape {content_shape}")

        if generated_image.shape != content_shape:
            raise TypeError(f"generated_image must be a tensor \
of shape {content_shape}")
        preprocessed = tf.keras.applications.vgg19.preprocess_input(generated_image * 255)
        generated_features = self.model(preprocessed)
        generated_style = generated_features[:-1]
        generated_content = generated_features[-1]

        content_cost = self.content_cost(generated_content)
        style_cost = self.style_cost(generated_style)
        total_cost = self.alpha * content_cost + self.beta * style_cost
        return total_cost, content_cost, style_cost