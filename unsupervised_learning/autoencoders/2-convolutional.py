#!/usr/bin/env python3

import tensorflow.keras as keras

def autoencoder(input_dims, filters, latent_dims):    
    Ei = keras.layers.Input(shape=input_dims)
    X = Ei
    for f in filters:
        X = keras.layers.Conv2D(f, (3, 3), activation='relu', padding='same')(X)
        X = keras.layers.MaxPooling2D((2, 2), padding='same')(X)
    Eo = X

    Di = keras.layers.Input(shape=latent_dims)
    X = Di
    for f in reversed(filters[1:]):
        X = keras.layers.Conv2D(f, (3, 3), activation='relu', padding='same')(X)
        X = keras.layers.UpSampling2D(size=(2, 2))(X)
    X = keras.layers.Conv2D(filters[0], (3, 3), activation='relu')(X)
    X = keras.layers.UpSampling2D(size=(2, 2))(X)
    Do = keras.layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid')(X)

    encoder = keras.Model(inputs=Ei, outputs=Eo)
    decoder = keras.Model(inputs=Di, outputs=Do)
    auto = keras.Model(inputs=Ei, outputs=decoder(encoder(Ei)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
