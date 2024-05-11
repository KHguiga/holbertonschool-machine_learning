#!/usr/bin/env python3
"""
   Mini-batch
"""

shuffle_data = __import__('2-shuffle_data').shuffle_data

def create_mini_batches(X, Y, batch_size):
    """
    Function to create mini-batches from input data X and labels Y.

    :param X: ndarray, input data of shape (m, nx)
    :param Y: ndarray, labels of shape (m, ny)
    :param batch_size: int, size of each mini-batch

    :return: list of mini-batches containing tuples (X_batch, Y_batch)
    """
    m = X.shape[0]
    # Shuffle the data
    X_shuffled, Y_shuffled = shuffle_data(X, Y)
    # calculate number of batches
    num_batches = m // batch_size + (m % batch_size != 0)
    # print("nbr_batch",nbr_batch)
    mini_batches = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, m)
        X_batch = X_shuffled[start_idx:end_idx]
        Y_batch = Y_shuffled[start_idx:end_idx]
        mini_batches.append((X_batch, Y_batch))
    return mini_batches
