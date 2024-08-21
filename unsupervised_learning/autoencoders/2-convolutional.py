#!/usr/bin/env python3
"""
Project auto encoders
Bu Ced+
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    cet autoencodeur utilise des filtres convolutif
    suive de maxpool pour le retrecissement et de upsampling
    La difficulté est dans le padding pour respecter les dimensions
    on se retrouve avec des dimensions impaires
    """

    # creation de l'encodeur, 1ere moitié
    encoder_input = keras.layers.Input(shape=input_dims)
    X = encoder_input 
    for n in filters:
        X = keras.layers.Conv2D(n,activation='relu', kernel_size=(3, 3),padding='same')(X)
        X = keras.layers.MaxPooling2D((2, 2), padding='same')(X)

    encoder_output = X
    encoder = keras.models.Model(encoder_input, encoder_output, name='encoder')
   

    # création du decoder et de la recontruction de l'image
    decoder_input = keras.layers.Input(shape=latent_dims)
    X = decoder_input

    for n in reversed(filters[1:]):
        X = keras.layers.Conv2D(n, activation='relu', padding='same', kernel_size=(3, 3))(X)
        X = keras.layers.UpSampling2D((2, 2))(X)

    X = keras.layers.Conv2D(filters[-1], activation='sigmoid', padding='valid',
                       kernel_size=(3, 3))(X)

    X = keras.layers.UpSampling2D((2, 2))(X)
    decoder_output = keras.layers.Conv2D(input_dims[2], activation='sigmoid', padding='same', kernel_size=(3, 3))(X)
    decoder = keras.models.Model(decoder_input, decoder_output, name='decoder')
    decoder.summary()
    # Full autoencoder

    encoded = encoder(encoder_input)
    decoded = decoder(encoded)
    auto = keras.models.Model(encoder_input, decoded, name='autoencoder')
    
    # Compile the autoencoder
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto