#!/usr/bin/env python3
"""Comment of Class"""

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    This class represents the decoder for machine translation
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Initializes the RNNDecoder.
        """
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units=units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """Forward pass for the RNN decoder with attention mechanism."""
        context, _ = self.attention(s_prev, hidden_states)

        x = self.embedding(x)

        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)

        output, s = self.gru(x)

        output = tf.squeeze(output, axis=1)

        y = self.F(output)

        return y, s
