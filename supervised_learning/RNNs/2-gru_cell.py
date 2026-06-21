#!/usr/bin/env python3
"""
Module GRUCell
Contains the class GRUCell for a single Gated Recurrent Unit cell.
"""

import numpy as np


class GRUCell:
    """
    Represents a single cell of a Gated Recurrent Unit (GRU) network.
    """

    def __init__(self, i, h, o):
        """Initializes the GRU cell parameters."""
        self.Wz = np.random.randn(i + h, h)
        self.bz = np.zeros((1, h))
        self.Wr = np.random.randn(i + h, h)
        self.br = np.zeros((1, h))
        self.Wh = np.random.randn(i + h, h)
        self.bh = np.zeros((1, h))
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Performs forward propagation for one time step of the GRU cell."""
        concat = np.concatenate((h_prev, x_t), axis=1)
        z_t = 1 / (1 + np.exp(-(np.matmul(concat, self.Wz) + self.bz)))
        r_t = 1 / (1 + np.exp(-(np.matmul(concat, self.Wr) + self.br)))
        concat_candidate = np.concatenate((r_t * h_prev, x_t), axis=1)
        h_tilde = np.tanh(np.matmul(concat_candidate, self.Wh) + self.bh)
        h_next = (1 - z_t) * h_prev + z_t * h_tilde
        y_linear = np.matmul(h_next, self.Wy) + self.by
        exp = np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))
        y = exp / np.sum(exp, axis=1, keepdims=True)
        return h_next, y
