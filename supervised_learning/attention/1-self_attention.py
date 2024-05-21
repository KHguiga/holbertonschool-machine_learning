#!/usr/bin/env python3
"""
    Module to create Class SelfAttention
"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
        class to calculate attention ofr machine translation
    """
    def __init__(self, units):
        """
            class constructor
        :param units: integer, number hidden units in alignment model
        """
        if not isinstance(units, int):
            raise TypeError("units should be an integer")

        self.W = tf.keras.layers.Dense(units=units)
        self.U = tf.keras.layers.Dense(units=units)
        self.V = tf.keras.layers.Dense(units=1)

    def call(self, s_prev, hidden_states):
        """
            call method

        :param s_prev: tensor, shape(batch, units) prev decoder hidden state
        :param hidden_states: tensor, shape(batch, input_seq_len, units),
            outputs of the encoder

        :return: context, weights
            context: tensor, shape(batch, units) context vector for decoder
            weights: tensor, shape(batch, input_seq_len, units),
                attention weights
        """
        # calculate score with previous decoder hidden state
        
        s_prev_score = tf.expand_dims(s_prev, axis=1)
        s_prev_score = self.W(s_prev_score)
        # calculate score with encoder hidden states
        encoder_score = self.U(hidden_states)

        # calculate attention scores
        scores = tf.nn.tanh(s_prev_score + encoder_score)
        scores = self.V(scores)

        # attention weights
        att_weights = tf.nn.softmax(scores, axis=1)

        # context vector
        context = att_weights * hidden_states
        context = tf.reduce_sum(context, axis=1)

        return context, att_weights
