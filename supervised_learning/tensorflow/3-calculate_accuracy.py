
#!/usr/bin/env python3
'''
Contains the method that calculates the accuracy 
of a prediction
'''
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):

    y_tensor = tf.argmax(y, axis=1)
    y_pred_tensor = tf.argmax(y_pred, axis=1)

    correct_predictions = tf.math.equal(y_tensor, y_pred_tensor)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    return accuracy