#!/usr/bin/env python3
"""Bidirectional RNN cell module."""
import numpy as np


class BidirectionalCell:
    """Represents a bidirectional RNN cell."""

    def __init__(self, i, h, o):
        """
        Initialize the bidirectional cell.

        Args:
            i (int): Dimensionality of the input data.
            h (int): Dimensionality of the hidden states.
            o (int): Dimensionality of the outputs.
        """
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(2 * h, o)

        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Calculate the next hidden state in the forward direction.

        Args:
            h_prev (np.ndarray): Previous hidden state of shape (m, h).
            x_t (np.ndarray): Input data of shape (m, i).

        Returns:
            np.ndarray: Next hidden state of shape (m, h).
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(concat, self.Whf) + self.bhf)
        return h_next
