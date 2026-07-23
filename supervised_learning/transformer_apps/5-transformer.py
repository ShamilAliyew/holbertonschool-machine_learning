#!/usr/bin/env python3
"""Transformer model for machine translation."""

import tensorflow as tf


def positional_encoding(max_seq_len, dm):
    """Calculate positional encodings."""
    positions = tf.cast(tf.range(max_seq_len)[:, tf.newaxis], tf.float32)
    dimensions = tf.cast(tf.range(dm)[tf.newaxis, :], tf.float32)
    exponents = 2 * tf.floor(dimensions / 2) / tf.cast(dm, tf.float32)
    angles = positions / tf.pow(10000.0, exponents)
    even_dimensions = tf.equal(tf.range(dm) % 2, 0)[tf.newaxis, :]

    return tf.where(even_dimensions, tf.sin(angles), tf.cos(angles))


def sdp_attention(Q, K, V, mask=None):
    """Calculate scaled dot-product attention."""
    scores = tf.matmul(Q, K, transpose_b=True)
    depth = tf.cast(tf.shape(K)[-1], tf.float32)
    scores /= tf.math.sqrt(depth)

    if mask is not None:
        scores += mask * -1e9

    weights = tf.nn.softmax(scores, axis=-1)

    return tf.matmul(weights, V), weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """Perform multi-head attention."""

    def __init__(self, dm, h):
        """Initialize the layer."""
        super(MultiHeadAttention, self).__init__()
        self.dm = dm
        self.h = h
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """Split the last dimension into attention heads."""
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))

        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask=None):
        """Perform the forward pass."""
        batch_size = tf.shape(Q)[0]
        Q = self.split_heads(self.Wq(Q), batch_size)
        K = self.split_heads(self.Wk(K), batch_size)
        V = self.split_heads(self.Wv(V), batch_size)

        attention, weights = sdp_attention(Q, K, V, mask)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        attention = tf.reshape(attention, (batch_size, -1, self.dm))

        return self.linear(attention), weights


class EncoderBlock(tf.keras.layers.Layer):
    """Represent one Transformer encoder block."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialize the encoder block."""
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training=False, mask=None):
        """Perform the forward pass."""
        attention, _ = self.mha(x, x, x, mask)
        attention = self.dropout1(attention, training=training)
        out1 = self.layernorm1(x + attention)

        output = self.dense_output(self.dense_hidden(out1))
        output = self.dropout2(output, training=training)

        return self.layernorm2(out1 + output)


class DecoderBlock(tf.keras.layers.Layer):
    """Represent one Transformer decoder block."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialize the decoder block."""
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training=False,
             look_ahead_mask=None, padding_mask=None):
        """Perform the forward pass."""
        attention1, _ = self.mha1(x, x, x, look_ahead_mask)
        attention1 = self.dropout1(attention1, training=training)
        out1 = self.layernorm1(x + attention1)

        attention2, _ = self.mha2(
            out1, encoder_output, encoder_output, padding_mask
        )
        attention2 = self.dropout2(attention2, training=training)
        out2 = self.layernorm2(out1 + attention2)

        output = self.dense_output(self.dense_hidden(out2))
        output = self.dropout3(output, training=training)

        return self.layernorm3(out2 + output)


class Encoder(tf.keras.layers.Layer):
    """Represent the Transformer encoder."""

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """Initialize the encoder."""
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [
            EncoderBlock(dm, h, hidden, drop_rate) for _ in range(N)
        ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training=False, mask=None):
        """Perform the forward pass."""
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len, :]
        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(x, training=training, mask=mask)

        return x


class Decoder(tf.keras.layers.Layer):
    """Represent the Transformer decoder."""

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """Initialize the decoder."""
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [
            DecoderBlock(dm, h, hidden, drop_rate) for _ in range(N)
        ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training=False,
             look_ahead_mask=None, padding_mask=None):
        """Perform the forward pass."""
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len, :]
        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(
                x,
                encoder_output,
                training=training,
                look_ahead_mask=look_ahead_mask,
                padding_mask=padding_mask
            )

        return x


class Transformer(tf.keras.Model):
    """Represent a complete Transformer network."""

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """Initialize the Transformer."""
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            N, dm, h, hidden, input_vocab, max_seq_input, drop_rate
        )
        self.decoder = Decoder(
            N, dm, h, hidden, target_vocab, max_seq_target, drop_rate
        )
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training=False, encoder_mask=None,
             look_ahead_mask=None, decoder_mask=None):
        """Perform the forward pass."""
        encoder_output = self.encoder(
            inputs,
            training=training,
            mask=encoder_mask
        )
        decoder_output = self.decoder(
            target,
            encoder_output,
            training=training,
            look_ahead_mask=look_ahead_mask,
            padding_mask=decoder_mask
        )

        return self.linear(decoder_output)
