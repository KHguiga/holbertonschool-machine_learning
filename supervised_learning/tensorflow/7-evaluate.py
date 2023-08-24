#!/usr/bin/env python3
"""evaluate neural network method"""
import tensorflow.compat.v1 as tf

def evaluate(X, Y, save_path):
    """Evaluates a neural network"""
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(save_path + ".meta")
        new_saver.restore(sess, tf.train.latest_checkpoint('./'))
        y_pred = tf.get_collection("y_pred")[0]
        loss = tf.get_collection("loss")[0]
        accuracy = tf.get_collection("accuracy")[0]
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        y_pred, accuracy, loss = sess.run([y_pred, accuracy, loss],
                                          feed_dict={x: X, y: Y})
    return y_pred, accuracy, loss