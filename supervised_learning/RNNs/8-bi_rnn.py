#!/usr/bin/env python3
"""Bidirectional RNN forward propagation module."""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Perform forward propagation for a bidirectional RNN.

    Args:
        bi_cell: Instance of BidirectionalCell used for propagation.
        X (np.ndarray): Input data of shape (t, m, i).
        h_0 (np.ndarray): Initial forward hidden state of shape (m, h).
        h_t (np.ndarray): Initial backward hidden state of shape (m, h).

    Returns:
        tuple: (H, Y)
            H (np.ndarray): Concatenated hidden states of shape (t, m, 2 * h).
            Y (np.ndarray): Outputs of shape (t, m, o).
    """
    t, m, _ = X.shape
    h = h_0.shape[1]

    Hf = np.zeros((t + 1, m, h))
    Hb = np.zeros((t + 1, m, h))

    Hf[0] = h_0
    for step in range(t):
        Hf[step + 1] = bi_cell.forward(Hf[step], X[step])

    Hb[t] = h_t
    for step in range(t - 1, -1, -1):
        Hb[step] = bi_cell.backward(Hb[step + 1], X[step])

    H = np.concatenate((Hf[1:], Hb[:-1]), axis=2)
    Y = bi_cell.output(H)

    return H, Y
