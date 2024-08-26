#!/usr/bin/env python3
"""This modlue contains a function that creates a variational autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):

    encoder_inputs = keras.Input(shape=(input_dims,))

    # Create the encoder layers
    for idx, units in enumerate(hidden_layers):
        # Add dense layers with the relu activation function
        layer = keras.layers.Dense(units=units, activation="relu")
        if idx == 0:
            # If it is the first layer, set the input
            outputs = layer(encoder_inputs)

        else:
            # If it is not the first layer, set the
            # output of the previous layer
            outputs = layer(outputs)

    # crerate a mean layer
    layer = keras.layers.Dense(units=latent_dims)

    # Create the mean layer
    mean = layer(outputs)

    layer = keras.layers.Dense(units=latent_dims)

    log_variation = layer(outputs)


    # Create a sampling function to sample from the mean and log variation
    def sampling(args):
        """This function samples from the mean and log variation
        Args:
            args: list containing the mean and log variation
        Returns: sampled tensor
        """
        # Get the mean and log variation from the arguments
        mean, log_variation = args

        # Generate a random tensor
        epsilon = keras.backend.random_normal(shape=keras.backend.shape(mean))

        # Return the sampled tensor
        return mean + keras.backend.exp(log_variation * 0.5) * epsilon

    # Use a keras layer to wrap the sampling function
    z = keras.layers.Lambda(sampling, output_shape=(latent_dims,))(
        [mean, log_variation]
    )

    # Create the encoder model
    encoder = keras.models.Model(
        inputs=encoder_inputs, outputs=[z, mean, log_variation]
    )

    # Define the decoder model
    decoder_inputs = keras.Input(shape=(latent_dims,))
    for idx, units in enumerate(reversed(hidden_layers)):
        # Create a Dense layer with relu activation
        layer = keras.layers.Dense(units=units, activation="relu")

        if idx == 0:
            # if it is the first layer, set the input
            outputs = layer(decoder_inputs)

        else:
            outputs = layer(outputs)

    # Create the output layer for the decoder modle
    # using sigmoid activation function
    layer = keras.layers.Dense(units=input_dims, activation="sigmoid")

    outputs = layer(outputs)

    # Create the decoder model
    decoder = keras.models.Model(inputs=decoder_inputs, outputs=outputs)

    # Create the full autoencoder model

    outputs = decoder(outputs[0])

    auto = keras.models.Model(inputs=encoder_inputs, outputs=outputs)

    reconstruction_loss = keras.losses.binary_crossentropy(
        encoder_inputs, outputs) * input_dims

    kl_loss = 1 + log_variation - \
        keras.backend.square(mean) - keras.backend.exp(log_variation)
    kl_loss = keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)

    auto.add_loss(vae_loss)
    auto.compile(optimizer='adam')

    return encoder, decoder, auto