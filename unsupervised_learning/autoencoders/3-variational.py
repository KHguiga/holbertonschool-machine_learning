#!/usr/bin/env python3
"""Variational Autoencoder Module"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a variational autoencoder:

    input_dims is an integer containing the dimensions of the model input
    hidden_layers is a list containing the number of nodes for each hidden
    layer in the encoder, respectively
    the hidden layers should be reversed for the decoder
    latent_dims is an integer containing the dimensions of the latent space
    representation

    Returns: encoder, decoder, auto
    encoder is the encoder model, which should output the latent
    representation, the mean, and the log variance, respectively
    decoder is the decoder model
    auto is the full autoencoder model
    The autoencoder model should be compiled using adam optimization and binary
    cross-entropy loss
    All layers should use a relu activation except for the mean and log
    variance layers in the encoder, which should use None, and the last layer
    in the decoder, which should use sigmoid
    """
    encoder_input = keras.layers.Input(shape=(input_dims,))
    encoder_output = encoder_input
    for units in hidden_layers:
        encoder_output = keras.layers.Dense(
            units, activation='relu')(encoder_output)

    mean = keras.layers.Dense(latent_dims)(encoder_output)
    log_sigma = keras.layers.Dense(latent_dims)(encoder_output)

    def sampling(args):
        z_mean, z_log_sigma = args
        m = keras.backend.shape(z_mean)[0]
        n = keras.backend.int_shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(m,n), mean=0., stddev=1.)
        return z_mean + keras.backend.exp(0.5* z_log_sigma) * epsilon

    latent_space = keras.layers.Lambda(sampling)([mean, log_sigma])
    encoder = keras.models.Model(
        encoder_input, [mean, log_sigma, latent_space])

    decoder_input = keras.layers.Input(shape=(latent_dims,))
    decoder_output = decoder_input
    for units in reversed(hidden_layers):
        decoder_output = keras.layers.Dense(
            units, activation='relu')(decoder_output)

    decoder_output = keras.layers.Dense(
        input_dims, activation='sigmoid')(decoder_output)
    decoder = keras.models.Model(decoder_input, decoder_output)

    autoencoder = keras.models.Model(
        encoder_input, decoder(encoder(encoder_input)[2]))

    reconstruction_loss = keras.losses.binary_crossentropy(
        encoder_input, decoder(encoder(encoder_input)[2])) * input_dims

    kl_loss = 1 + log_sigma - \
        keras.backend.square(mean) - keras.backend.exp(log_sigma)
    kl_loss = keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)

    autoencoder.add_loss(vae_loss)
    autoencoder.compile(optimizer='adam')

    return encoder, decoder, autoencoder
