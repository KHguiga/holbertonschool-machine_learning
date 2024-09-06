import numpy as np


class GRUCell:
    """This class represents a GRU unit"""
    def __init__(self, i, h, o):

        self.Wz = np.random.normal(size=(i + h, h))
        self.bz = np.zeros((1, h))

        self.Wr = np.random.normal(size=(i + h, h))
        self.br = np.zeros((1, h))

        
        self.Wh = np.random.normal(size=(i + h, h))
        self.bh = np.zeros((1, h))

        
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        

        inputs = np.concatenate((h_prev, x_t), axis=1)

        reset = self.sigmoid(np.matmul(inputs, self.Wr) + self.br)

        update = self.sigmoid(np.matmul(inputs, self.Wz) + self.bz)

        updated_input = np.concatenate((reset * h_prev, x_t), axis=1)

        h_r = np.tanh(np.matmul(updated_input, self.Wh) + self.bh)

        h_next = update * h_r + (1 - update) * h_prev

        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, y

    def sigmoid(self, x):
        
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
