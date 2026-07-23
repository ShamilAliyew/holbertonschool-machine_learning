#!/usr/bin/env python3
"""Creates bag-of-words embeddings."""

from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """Create a bag-of-words embedding matrix.

    Args:
        sentences: A list of sentences to analyze.
        vocab: An optional list of vocabulary words to use.

    Returns:
        A tuple containing the embedding matrix and the analyzed features.
    """
    vectorizer = CountVectorizer(vocabulary=vocab)
    embeddings = vectorizer.fit_transform(sentences).toarray()
    features = vectorizer.get_feature_names_out()

    return embeddings, features
