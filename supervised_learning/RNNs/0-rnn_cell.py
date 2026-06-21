#!/usr/bin/env python3
"""
Module RNNs
Contains the class RNNCell for a simple Recurrent Neural Network cell.
"""

import numpy as np


class RNNCell:
    """
    Represents a single cell of a vanilla (simple) Recurrent Neural Network.
    """

    def __init__(self, i, h, o):
        """Initializes the RNN cell parameters."""
        self.Wh = np.random.randn(i + h, h)
        self.bh = np.zeros((1, h))
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Performs forward propagation for a single time step."""
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(concat, self.Wh) + self.bh)
        y_linear = np.matmul(h_next, self.Wy) + self.by
        exp = np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))
        y = exp / np.sum(exp, axis=1, keepdims=True)
        return h_next, y
