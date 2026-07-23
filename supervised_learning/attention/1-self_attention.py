#!/usr/bin/env python3
"""Comment of Class"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Self Attention class to calculate
       attention for machine translation"""

    def __init__(self, units):
        """
        Initialize the Self Attention layer
        """
        super(SelfAttention, self).__init__()

        self.W = tf.keras.layers.Dense(units)

        self.U = tf.keras.layers.Dense(units)

        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """Calculate the context vector and attention weights"""
        s_prev_expanded = tf.expand_dims(s_prev, 1)

        W_s = self.W(s_prev_expanded)

        U_h = self.U(hidden_states)

        tanh_sum = tf.nn.tanh(W_s + U_h)

        score = self.V(tanh_sum)

        weights = tf.nn.softmax(score, axis=1)

        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
