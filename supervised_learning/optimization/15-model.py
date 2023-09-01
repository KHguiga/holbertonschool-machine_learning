#!/usr/bin/env python3
'''
Moulus that builds, tranis and saves a NN
'''
import numpy as np
import tensorflow.compat.v1 as tf


def shuffle_data(X, Y):
    '''Shuffle data X, Y'''
    m = X.shape[0]
    shuf_vect = list(np.random.permutation(m))
    x = X[shuf_vect, :]
    y = Y[shuf_vect, :]
    return x, y

def create_layer(prev, n, activation):
    '''Function that creates a layer'''
    activa = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=activa, name='layer')
    return layer(prev)


def create_batch_norm_layer(prev, n, activation):
    """
    apply the activation function to the normalized inputs
    """
    z = create_layer(prev, n, activation)
    if activation is None:
        return z
    else:
        mean, variance = tf.nn.moments(z, axes=[0])
        gamma = tf.Variable(1., trainable=True)
        beta = tf.Variable(0., trainable=True)
        epsilon = 1e-8
        z_norm = tf.nn.batch_normalization(
            z, mean, variance, beta, gamma, epsilon)
    return activation(z_norm)

def forward_prop(prev, layers, activations, epsilon):
    #all layers get batch_normalization but the last one, that stays without any activation or normalization
    for i, n in enumerate(layers[:-1]):
        prev = create_batch_norm_layer(prev, n, activations[i])
    prev = create_layer(prev, layers[-1],activations[-1])
    return prev

def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    '''Function that calculates learning rate decay'''
    learning = tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                           decay_rate, staircase=True)
    return learning


def create_Adam_op(loss, alpha, beta1, beta2, epsilon, global_step):
    '''Fucntion that calculates Adam'''
    adam = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    return adam.minimize(loss, global_step=global_step)

def calculate_accuracy(y, y_pred):
    '''Function that calculates accuracy'''
    yes_not = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    acura = tf.reduce_mean(tf.cast(yes_not, tf.float32))
    return acura


def calculate_loss(y, y_pred):
    '''Fuction that calculates loss'''
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    return loss


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    '''Function that trains NN'''
    nx = Data_train[0].shape[1]
    classes = Data_train[1].shape[1]
    (X_train, Y_train) = Data_train
    (X_valid, Y_valid) = Data_valid
    x = tf.placeholder(tf.float32, shape=[None, nx], name="x")
    tf.add_to_collection("x", x)
    y = tf.placeholder(tf.float32, shape=[None, classes], name="y")
    tf.add_to_collection("y", y)
    y_pred = forward_prop(x, layers, activations)
    tf.add_to_collection("y_pred", y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection("accuracy", accuracy)
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection("loss", loss)
    global_step = tf.Variable(0, trainable=False)
    m = X_train.shape[0]
    decay_steps = m // batch_size
    if m % batch_size:
        decay_steps += 1
    alpha_d = learning_rate_decay(alpha, decay_rate, global_step, decay_steps)
    train_op = create_Adam_op(loss, alpha_d, beta1, beta2, epsilon, global_step)
    tf.add_to_collection("train_op", train_op)
    m = X_train.shape[0]
    if m % batch_size == 0:
        complete = 1
        num_batches = int(m / batch_size)
    else:
        complete = 0
        num_batches = int(m / batch_size) + 1
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        
        for i in range(epochs):
            print('After {} epochs:'.format(i))
            train_cost, train_accuracy = sess.run((loss, accuracy),
                                                  feed_dict={x:X_train,
                                                             y:Y_train})
            print('\tTraining Cost: {}'.format(train_cost))
            print('\tTraining Accuracy: {}'.format(train_accuracy))
            valid_cost, valid_accuracy = sess.run((loss, accuracy),
                                                  feed_dict={x:X_valid,
                                                             y:Y_valid})
            print('\tValidation Cost: {}'.format(valid_cost))
            print('\tValidation Accuracy: {}'.format(valid_accuracy))
            X_shuffle, Y_shuffle = shuffle_data(X_train, Y_train)
            for j in range(0, X_train.shape[0], batch_size):
                X_batch = X_shuffle[j:j + batch_size]
                Y_batch = Y_shuffle[j:j + batch_size]
                sess.run(train_op, feed_dict={x:X_batch, y:Y_batch})
                if not ((j // batch_size + 1) % 100):
                    cost, acc = sess.run((loss, accuracy), feed_dict={x:X_batch, y:Y_batch})
                    print('\tStep {}:'.format(j // batch_size + 1))
                    print('\t\tCost: {}'.format(cost))
                    print('\t\tAccuracy: {}'.format(acc))

        print('After {} epochs:'.format(epochs))
        train_cost, train_accuracy = sess.run((loss, accuracy),
                                              feed_dict={x:X_train, y:Y_train})
        print('\tTraining Cost: {}'.format(train_cost))
        print('\tTraining Accuracy: {}'.format(train_accuracy))
        valid_cost, valid_accuracy = sess.run((loss, accuracy),
                                              feed_dict={x:X_valid, y:Y_valid})
        print('\tValidation Cost: {}'.format(valid_cost))
        print('\tValidation Accuracy: {}'.format(valid_accuracy))

        saver = tf.train.Saver()
        return saver.save(sess, save_path)