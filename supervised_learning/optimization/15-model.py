#!/usr/bin/env python3

import numpy as np
import tensorflow.compat.v1 as tf

def shuffle_data(X, Y):
    p = np.random.permutation(X.shape[0])
    return X[p], Y[p]

def forward_prop(x, layers, activations, epsilon):
    prev = x
    for i, n in enumerate(layers):
        dense = tf.layers.Dense(n, kernel_initializer=tf.keras.initializers.VarianceScaling(mode='fan_avg'), name='dense')
        z = dense(prev)
        if i < len(layers) - 1:
            gamma = tf.Variable(tf.constant(1.0, shape=[n]), name='gamma',
                                trainable=True)
            beta = tf.Variable(tf.constant(0.0, shape=[n]), name='beta',
                               trainable=True)
            m, v = tf.nn.moments(z, axes=0)
            z_norm = tf.nn.batch_normalization(z, m, v, beta, gamma, epsilon)
            activation = activations[i]
            prev = activation(z_norm)
        else:
            prev = z
    return prev

def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    m, nx = X_train.shape
    classes = Y_train.shape[1]

    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    y_pred = forward_prop(x, layers, activations, epsilon)
    tf.add_to_collection('y_pred', y_pred)

    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    tf.add_to_collection('loss', loss)

    correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.add_to_collection('accuracy', accuracy)

    global_step = tf.Variable(0, trainable=False)
    decay_steps = m // batch_size
    if m % batch_size:
        decay_steps += 1

    alpha = tf.train.inverse_time_decay(alpha, global_step, decay_steps,
                                        decay_rate, staircase=True)
    optimizer = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    train_op = optimizer.minimize(loss, global_step=global_step)
    tf.add_to_collection('train_op', train_op)

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