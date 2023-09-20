#!/usr/bin/env python3

import tensorflow.keras as K

def build_model(nx, layers, activations, lambtha, keep_prob):
    if len(layers) != len(activations):
        raise ValueError("layers and activations must have the same number.")
    
    # Create a Sequential model
    model = K.Sequential()
    
    # Add the input layer
    model.add(K.layers.Dense(units=layers[0], input_dim=nx,
                            activation=activations[0],
                            kernel_regularizer=K.regularizers.l2(lambtha)))
    
    # Add hidden layers
    for i in range(1, len(layers)):
        model.add(K.layers.Dropout(1 - keep_prob))
        model.add(K.layers.Dense(units=layers[i], activation=activations[i],
                                kernel_regularizer=K.regularizers.l2(lambtha)))
    
    return model
