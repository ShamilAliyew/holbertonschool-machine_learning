#!/usr/bin/env python3
"""Deep RNN forward propagation module."""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Perform forward propagation for a deep (stacked) RNN.

    Args:
        rnn_cells (list): List of RNNCell instances of length l.
        X (np.ndarray): Input data of shape (t, m, i).
        h_0 (np.ndarray): Initial hidden state of shape (l, m, h).

    Returns:
        tuple: (H, Y)
            H (np.ndarray): Hidden states of shape (t + 1, l, m, h).
            Y (np.ndarray): Outputs of shape (t, m, o).
    """
    t, m, _ = X.shape
    l, _, h = h_0.shape
    o = rnn_cells[-1].by.shape[1]

    H = np.zeros((t + 1, l, m, h))
    Y = np.zeros((t, m, o))

    H[0] = h_0

    for step in range(t):
        x = X[step]
        for layer in range(l):
            h_prev = H[step, layer]
            h_next, y = rnn_cells[layer].forward(h_prev, x)
            H[step + 1, layer] = h_next
            x = h_next
        Y[step] = y

    return H, Y
