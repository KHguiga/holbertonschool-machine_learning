#!/usr/bin/env python3
"""This module contains a function that
builds a modified version of the LeNet-5 architecture using TensorFlow"""
import tensorflow.compat.v1 as tf  # type: ignore

# tf.disable_v2_behavior()


def lenet5(x, y):
    """
    Function that builds a modified version of the
    LeNet-5 architecture using TensorFlow

    Parameters:
    x is a tf.placeholder of shape (m, 28, 28, 1)
    containing the input images for the network
    y is a tf.placeholder of shape (m, 10) containing
    the one-hot labels for the network

    Returns:
    a tensor for the softmax activated output
    a training operation that utilizes Adam optimization (
        with default hyperparameters)
    a tensor for the loss of the network
    a tensor for the accuracy of the network
    """
    init = tf.keras.initializers.VarianceScaling(scale=2.0)
    relu = tf.nn.relu

    # Convolutional layer with 6 kernels of shape 5x5 with same padding
    conv1 = tf.layers.conv2d(
        x,
        filters=6,
        kernel_size=5,
        padding="same",
        activation=relu,
        kernel_initializer=init,
    )

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2)

    # Convolutional layer with 16 kernels of shape 5x5 with valid padding
    conv2 = tf.layers.conv2d(
        pool1,
        filters=16,
        kernel_size=5,
        padding="valid",
        activation=relu,
        kernel_initializer=init,
    )

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2)

    # Flatten the pool2 output
    flat = tf.layers.flatten(pool2)

    # Fully connected layer with 120 nodes
    fc1 = tf.layers.dense(
        flat, units=120, activation=relu, kernel_initializer=init
    )

    # Fully connected layer with 84 nodes
    fc2 = tf.layers.dense(fc1, units=84,
                          activation=relu, kernel_initializer=init)


    FC3 = tf.layers.Dense(10, kernel_initializer=init)
    A8 = FC3(fc2)
    softmax = tf.nn.softmax(A8)
    loss = tf.losses.softmax_cross_entropy(y, softmax)
    # Fully connected softmax output layer with 10 nodes
    # softmax = tf.layers.dense(
    #     fc2, units=10, activation=tf.nn.softmax, kernel_initializer=init
    # )

    # Loss
    # loss = tf.losses.softmax_cross_entropy(y, softmax)

    # Training operation
    train_op = tf.train.AdamOptimizer().minimize(loss)

    # Accuracy
    correct_prediction = tf.equal(tf.argmax(softmax, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return softmax, train_op, loss, accuracy