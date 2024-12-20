#!/usr/bin/env python3
""" This module defines the convolutional_GenDiscr function. """
import tensorflow as tf
from tensorflow import keras


def convolutional_GenDiscr():
    """
    This function instantiates a generator and discriminator models for a GAN.
    Returns:
        The generator and discriminator models.
    """
    def generator():
        """
        This function returns a generator model.
        Returns:
            The generator model.
        """
        # Input layer: 16 features
        input_layer = keras.layers.Input(shape=(16,))

        # Fully connected layer to expand to 2048 features
        x = keras.layers.Dense(2048)(input_layer)
        # Reshape into a 4D tensor: (2, 2, 512)
        x = keras.layers.Reshape((2, 2, 512))(x)

        # Upsampling to 4x4
        x = keras.layers.UpSampling2D()(x)
        # Convolutional layer to reduce the number of filters
        x = keras.layers.Conv2D(64, (3, 3), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)  # Normalize for stable training
        x = keras.layers.Activation('tanh')(x)    # Apply 'tanh' activation

        # Upsampling to 8x8
        x = keras.layers.UpSampling2D()(x)
        # Another convolutional layer
        x = keras.layers.Conv2D(16, (3, 3), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('tanh')(x)

        # Upsampling to 16x16
        x = keras.layers.UpSampling2D()(x)
        # Final convolutional layer to output 1 channel
        x = keras.layers.Conv2D(1, (3, 3), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        # Final activation to normalize outputs
        output_layer = keras.layers.Activation('tanh')(x)

        # Return the generator model
        return keras.Model(inputs=input_layer,
                     outputs=output_layer, name="generator")

    def get_discriminator():
        """
        This function returns a discriminator model.
        Returns:
            The discriminator model.
        """
        # Input layer: image of shape (16, 16, 1)
        input_layer = keras.layers.Input(shape=(16, 16, 1))

        # First convolutional layer
        x = keras.layers.Conv2D(32, (3, 3), padding='same')(input_layer)
        # Downsampling to reduce spatial dimensions
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation('tanh')(x)

        # Second convolutional layer
        x = keras.layers.Conv2D(64, (3, 3), padding='same')(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation('tanh')(x)

        # Third convolutional layer
        x = keras.layers.Conv2D(128, (3, 3), padding='same')(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation('tanh')(x)

        # Fourth convolutional layer
        x = keras.layers.Conv2D(256, (3, 3), padding='same')(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation('tanh')(x)

        # Flatten the 4D tensor to 1D for the final dense layer
        x = keras.layers.Flatten()(x)
        # Fully connected layer to output a single value (real or fake)
        output_layer = keras.layers.Dense(1)(x)

        # Return the discriminator model
        return keras.Model(inputs=input_layer,
                     outputs=output_layer, name="discriminator")

    return generator(), get_discriminator()
