#!/usr/bin/env python3
"""This module contains the GRUCell class"""
import numpy as np


class GRUCell:
    """This class represents a GRU unit"""
    def __init__(self, i, h, o):
        """Class constructor
        Args:
            i: is the dimensionality of the data
            h: is the dimensionality of the hidden state
            o: is the dimensionality of the outputs
        """
        # initialize Wz and bz
        # Wz shape (i + h, h) and bz shape (1, h)
        # Wz: This weight matrix is used to combine the input data and
        # the previous hidden state
        # to compute the new hidden state. The new hidden state is then
        # used to compute the output.
        # This is the update gate weight matrix.
        self.Wz = np.random.normal(size=(i + h, h))
        self.bz = np.zeros((1, h))

        # Initialize Wr and br
        # Wr: Weight matrix for the reset gate,
        # combining input data and previous hidden state
        # br: Bias for the reset gate
        self.Wr = np.random.normal(size=(i + h, h))
        self.br = np.zeros((1, h))

        # Initialize Wh and bh
        # Wh: Weight matrix for the candidate hidden state,
        # combining input data and previous hidden state
        # bh: Bias for the candidate hidden state
        self.Wh = np.random.normal(size=(i + h, h))
        self.bh = np.zeros((1, h))

        # Initialize Wy and by
        # Wy: Weight matrix for the output, combining the
        # hidden state to produce the output
        # by: Bias for the output
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """This method calculates the forward propagation for one time step
        Args:
            h_prev: numpy.ndarray of shape (m, h) containing the previous
                    hidden state
            x_t: numpy.ndarray of shape (m, i) that contains the data input
                 for the cell
        Returns: h_next, y
                 h_next: the next hidden state
                 y: the output of the cell
        """
        # Concatenate the previous hidden state (h_prev) and the input
        # data (x_t)
        # This combined input will be used to compute the gates and the
        # candidate hidden state
        # axis=1 to concatenate horizontally
        inputs = np.concatenate((h_prev, x_t), axis=1)

        # Create the update gate and reset gate
        # The update gate (z_t) determines how much of the previous
        # hidden state to retain
        update = self.sigmoid(np.matmul(inputs, self.Wz) + self.bz)

        # The reset gate (r_t) determines how much of the previous
        # hidden state to forget
        reset = self.sigmoid(np.matmul(inputs, self.Wr) + self.br)

        # Update the inputs (h_prev) with the (rest) reset gate
        # and concatenate with the input data (x_t)
        # axis=1 to concatenate horizontally
        updated_input = np.concatenate((reset * h_prev, x_t), axis=1)

        # Compute the new hidden state of the cell
        # h_r is shape (m, h)
        h_r = np.tanh(np.matmul(updated_input, self.Wh) + self.bh)

        # Calculate the new hidden state of the cell
        # after factroing in the update gate
        # h_next is shape (m, h)
        h_next = update * h_r + (1 - update) * h_prev

        # Calculate the output of the cell
        # taking in account the new hidden state
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, y

    def sigmoid(self, x):
        """This method calculates the sigmoid function
        Args:
            x: numpy.ndarray
        Returns: the sigmoid function of x
        """
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """This method calculates the softmax function
        Args:
            x: numpy.ndarray
        Returns: the softmax function of x
        """
        max_x = np.amax(x, 1).reshape(x.shape[0], 1)  # Get the row-wise maximum
        e_x = np.exp(x - max_x)  # For stability
        return e_x / e_x.sum(axis=1, keepdims=True)