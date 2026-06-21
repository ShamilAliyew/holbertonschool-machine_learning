#!/usr/bin/env python3
"""LSTM cell module."""
import numpy as np


class LSTMCell:
    """Represents an LSTM unit."""

    def __init__(self, i, h, o):
        """
        Initialize the LSTM cell.

        Args:
            i (int): Dimensionality of the input data.
            h (int): Dimensionality of the hidden state.
            o (int): Dimensionality of the output.
        """
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def sigmoid(x):
        """Compute the sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        """Compute softmax with numerical stability."""
        x_shift = x - np.max(x, axis=1, keepdims=True)
        exp = np.exp(x_shift)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def forward(self, h_prev, c_prev, x_t):
        """
        Perform forward propagation for one time step.

        Args:
            h_prev (np.ndarray): Previous hidden state of shape (m, h).
            c_prev (np.ndarray): Previous cell state of shape (m, h).
            x_t (np.ndarray): Input data at time step t of shape (m, i).

        Returns:
            tuple: (h_next, c_next, y)
                h_next (np.ndarray): Next hidden state of shape (m, h).
                c_next (np.ndarray): Next cell state of shape (m, h).
                y (np.ndarray): Output of shape (m, o).
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        f_gate = self.sigmoid(np.matmul(concat, self.Wf) + self.bf)
        u_gate = self.sigmoid(np.matmul(concat, self.Wu) + self.bu)
        c_tilde = np.tanh(np.matmul(concat, self.Wc) + self.bc)
        o_gate = self.sigmoid(np.matmul(concat, self.Wo) + self.bo)

        c_next = f_gate * c_prev + u_gate * c_tilde
        h_next = o_gate * np.tanh(c_next)

        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, c_next, y
