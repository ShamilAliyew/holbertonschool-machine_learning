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

    def backward(self, h_next, x_t):
        """
        Calculate the previous hidden state in the backward direction.

        Args:
            h_next (np.ndarray): Next hidden state of shape (m, h).
            x_t (np.ndarray): Input data of shape (m, i).

        Returns:
            np.ndarray: Previous hidden state of shape (m, h).
        """
        concat = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.matmul(concat, self.Whb) + self.bhb)
        return h_prev

    @staticmethod
    def softmax(x):
        """Compute softmax with numerical stability."""
        x_shift = x - np.max(x, axis=-1, keepdims=True)
        exp = np.exp(x_shift)
        return exp / np.sum(exp, axis=-1, keepdims=True)

    def output(self, H):
        """
        Calculate all outputs for the bidirectional RNN.

        Args:
            H (np.ndarray): Concatenated hidden states of shape (t, m, 2 * h).

        Returns:
            np.ndarray: Outputs of shape (t, m, o).
        """
        logits = np.matmul(H, self.Wy) + self.by
        Y = self.softmax(logits)
        return Y
