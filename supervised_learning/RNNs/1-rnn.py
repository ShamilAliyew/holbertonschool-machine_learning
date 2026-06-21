#!/usr/bin/env python3
"""
Module rnn
Performs forward propagation for a simple RNN over all time steps.
"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN.
    """
    t, m, _ = X.shape
    h = h_0.shape[1]
    o = rnn_cell.by.shape[1]

    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))

    H[0] = h_0

    for step in range(t):
        h_next, y = rnn_cell.forward(H[step], X[step])
        H[step + 1] = h_next
        Y[step] = y

    return H, Y
