#!/usr/bin/env python3
"""Convert gensim Word2Vec embeddings to a Keras layer."""

import tensorflow as tf


def gensim_to_keras(model):
    """Convert a gensim Word2Vec model to a trainable Keras Embedding."""
    embedding_matrix = model.wv.vectors
    vocab_size, embedding_dim = embedding_matrix.shape

    layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        trainable=True
    )

    return layer
