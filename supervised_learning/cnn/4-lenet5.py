#!/usr/bin/env python3

import tensorflow.compat.v1 as tf

def lenet5(x, y):
    he_normal = tf.keras.initializers.VarianceScaling(scale=2.0)
    relu = tf.nn.relu

    C1 = tf.layers.Conv2D(6, 5, padding='same', activation=relu, kernel_initializer=he_normal)
    A1 = C1(x)
    P = tf.layers.MaxPooling2D(2, 2)
    A2 = P(A1)
    C3 = tf.layers.Conv2D(16, 5, activation=relu, kernel_initializer=he_normal)
    A3 = C3(A2)
    A4 = P(A3)
    F = tf.layers.Flatten()
    A5 = F(A4)
    FC1 = tf.layers.Dense(120, activation=relu, kernel_initializer=he_normal)
    A6 = FC1(A5)
    FC2 = tf.layers.Dense(84, activation=relu, kernel_initializer=he_normal)
    A7 = FC2(A6)
    FC2 = tf.layers.Dense(10, kernel_initializer=he_normal)
    A8 = FC2(A7)
    y_pred = tf.nn.softmax(A8)
    loss = tf.losses.softmax_cross_entropy(y, A8)
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    acc = tf.reduce_mean(tf.cast(correct, tf.float32))
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)

    return y_pred, train_op, loss, acc
