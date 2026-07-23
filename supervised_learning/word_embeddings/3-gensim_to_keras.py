#!/usr/bin/env python3
"""Converts a gensim Word2Vec model to a Keras Embedding layer."""

import tensorflow as tf


def gensim_to_keras(model):
    """Convert a trained gensim Word2Vec model to a trainable Keras layer.

    Args:
        model: A trained gensim Word2Vec model.

    Returns:
        A trainable Keras Embedding layer.
    """
    embedding = tf.keras.layers.Embedding(
        input_dim=len(model.wv.index_to_key),
        output_dim=model.wv.vector_size,
        embeddings_initializer=tf.keras.initializers.Constant(
            model.wv.vectors
        ),
        trainable=True
    )
    embedding.build((None,))

    return embedding
