#!/usr/bin/env python3
"""Creates and trains a Word2Vec model."""

import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """Create, build, and train a gensim Word2Vec model.

    Args:
        sentences: A list of tokenized sentences to train on.
        vector_size: The dimensionality of the word vectors.
        min_count: The minimum number of occurrences for included words.
        window: The maximum distance between current and predicted words.
        negative: The number of negative samples.
        cbow: If True, use CBOW; otherwise, use Skip-gram.
        epochs: The number of training iterations.
        seed: The random number generator seed.
        workers: The number of worker threads.

    Returns:
        The trained Word2Vec model.
    """
    model = gensim.models.Word2Vec(
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=not cbow,
        seed=seed,
        workers=workers,
        sorted_vocab=0
    )
    model.build_vocab(sentences)
    model.wv.sort_by_descending_frequency()
    model.make_cum_table()
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=epochs
    )

    return model
