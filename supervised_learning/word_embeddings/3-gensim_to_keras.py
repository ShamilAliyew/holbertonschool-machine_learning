#!/usr/bin/env python3
"""Converts a gensim Word2Vec model to a Keras Embedding layer."""


def gensim_to_keras(model):
    """Convert a trained gensim Word2Vec model to a trainable Keras layer.

    Args:
        model: A trained gensim Word2Vec model.

    Returns:
        A trainable Keras Embedding layer.
    """
    return model.wv.get_keras_embedding(train_embeddings=True)
