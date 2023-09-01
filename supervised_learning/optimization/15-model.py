#!/usr/bin/env python3
"""module which contain model function"""
import numpy as np
import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    """
    creates a batch normalization layer for a neural network in tensorflow:

    * prev is the activated output of the previous layer
    * n is the number of nodes in the layer to be created
    * activation is the activation function that should be
      used on the output of the layer
    * you should use the tf.keras.layers.Dense layer as the base
      layer with kernal initializer
      tf.keras.initializers.VarianceScaling(mode='fan_avg')
    * your layer should incorporate two trainable parameters, gamma and beta,
      initialized as vectors of 1 and 0 respectively
    * you should use an epsilon of 1e-8
    Returns: a tensor of the activated output for the layer
    """
    #  implement He-et-al initialization for the layer weights
    het_et_al = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    # kernel_initializer=het_et_al
    linear_model = tf.layers.Dense(name="layer",
                                   units=n,
                                   activation=None,
                                   kernel_initializer=het_et_al)
    layer = linear_model(prev)

    beta = tf.Variable(initial_value=tf.constant(0.0, shape=[n]),
                       name='beta')
    gamma = tf.Variable(initial_value=tf.constant(1.0, shape=[n]),
                        name='gamma')

    mean, std = tf.nn.moments(layer, axes=0, keep_dims=True)
    v1 = tf.nn.batch_normalization(layer, mean, std, offset=beta,
                                   scale=gamma, variance_epsilon=1e-8)
    return activation(v1)

def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    creates a learning rate decay operation
    in tensorflow using inverse time decay:

    * alpha is the original learning rate
    * decay_rate is the weight used to determine the rate
      at which alpha will decay
    * global_step is the number of passes of gradient
      descent that have elapsed
    * decay_step is the number of passes of gradient
      descent that should occur before alpha is decayed further
    * the learning rate decay should occur in a stepwise fashion
    Returns: the learning rate decay operation
    """
    return tf.compat.v1.train.inverse_time_decay(alpha, global_step,
                                                 decay_step, decay_rate,
                                                 staircase=True)

def forward_prop(prev, layers, activations, epsilon):
    #all layers get batch_normalization but the last one, that stays without any activation or normalization
    for size, activation in zip(layers, activations):
        prev = create_batch_norm_layer(prev, size, activation)
    return prev


def shuffle_data(X, Y):
    r_permuted = np.random.permutation(X.shape[0])
    return X[r_permuted], Y[r_permuted]

def calculate_accuracy(y, y_pred):
    """
    that calculates the accuracy of a prediction:

    y is a placeholder for the labels of the input data
    y_pred is a tensor containing the network’s predictions
    Returns: a tensor containing the decimal accuracy of the prediction
    """
    # returns the index with the largest value across axes of a tensor
    y_pred = tf.argmax(input=y_pred, axis=1)
    y = tf.argmax(input=y, axis=1)

    # returns the truth value of (y_pred == y) element-wise.
    equal = tf.equal(y, y_pred)

    # casting to avoid this error
    # **TypeError: Value passed to parameter 'input'
    # has DataType bool not in list of allowed values**
    cast = tf.cast(equal, dtype=tf.float32)
    return tf.reduce_mean(cast)

def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    creates the training operation for a neural network in tensorflow
    using the Adam optimization algorithm:

    * loss is the loss of the network
    * alpha is the learning rate
    * beta1 is the weight used for the first moment
    * beta2 is the weight used for the second moment
    * epsilon is a small number to avoid division by zero
    Returns: the Adam optimization operation
    """
    Adam = tf.train.AdamOptimizer(learning_rate=alpha,
                                  beta1=beta1, beta2=beta2,
                                  epsilon=epsilon)
    return Adam.minimize(loss)

def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    # get X_train, Y_train, X_valid, and Y_valid from Data_train and Data_valid
    X_train = Data_train[0]
    Y_train = Data_train[1]
    X_valid = Data_valid[0]
    Y_valid = Data_valid[1]

    # initialize x, y and add them to collection 
    x = tf.placeholder(tf.float32, shape=(None, X_train.shape[1]), name='x')
    tf.add_to_collection('x', x)
    y = tf.placeholder(tf.float32, shape=(None, Y_train.shape[1]), name='y')
    tf.add_to_collection('y', y)

    # initialize y_pred and add it to collection
    y_pred = forward_prop(x, layers, activations, epsilon)
    tf.add_to_collection('y_pred', y_pred)

    # intialize loss and add it to collection
    loss = tf.compat.v1.losses.softmax_cross_entropy(y, y_pred)
    tf.add_to_collection('loss', loss)

    # intialize accuracy and add it to collection
    accuracy =  calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)

    # intialize global_step variable
    # hint: not trainable
    global_step = tf.Variable(0, trainable=False)
    # compute decay_steps
    decay_steps = 10
    # create "alpha" the learning rate decay operation in tensorflow
    alpha = learning_rate_decay(alpha, decay_rate, global_step, decay_steps)

    # initizalize train_op and add it to collection 
    # hint: don't forget to add global_step parameter in optimizer().minimize()
    train_op = create_Adam_op(alpha, beta1, beta2, epsilon)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        m = X_train.shape[0]
        # instance of tf.train.Saver() to save
        saver = tf.train.Saver()

        for i in range(epochs + 1):
            # print training and validation cost and accuracy
            t_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            t_precision = sess.run(accuracy,
                                   feed_dict={x: X_train, y: Y_train})
            v_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            v_precision = sess.run(accuracy,
                                   feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(t_cost))
            print("\tTraining Accuracy: {}".format(t_precision))
            print("\tValidation Cost: {}".format(v_cost))
            print("\tValidation Accuracy: {}".format(v_precision))

            # shuffle data
            shuffled_X, shuffled_Y = shuffle_data(X_train, Y_train)

            for j in range(0, X_train.shape[0], batch_size):
                # get X_batch and Y_batch from X_train shuffled and Y_train shuffled
                start = j * batch_size
                end = (j + 1) * batch_size
                end = min(m, end)
                X_mini_batch = shuffled_X[start:end]
                Y_mini_batch = shuffled_Y[start:end]

                # run training operation
                sess.run(train_op, feed_dict={x: X_mini_batch,
                                              y: Y_mini_batch})

                if (j + 1) % 100 == 0 and j != 0:
                    t_cost = sess.run(loss, feed_dict={x: X_mini_batch,
                                                       y: Y_mini_batch})
                    t_precision = sess.run(accuracy,
                                            feed_dict={x: X_mini_batch,
                                                       y: Y_mini_batch})
                    # print training and validation cost and accuracy again
                    # print batch cost and accuracy
                    print("\tStep {}:".format(j + 1))
                    print("\t\tCost: {}".format(t_cost))
                    print("\t\tAccuracy: {}".format(t_precision))




        # save and return the path to where the model was saved
        return saver.save(sess, save_path)
