#!/usr/bin/env python3
"""Creates and trains a FastText model."""

from gensim.models import FastText


def fasttext_model(sentences, vector_size=100, min_count=5, negative=5,
                   window=5, cbow=True, epochs=5, seed=0, workers=1):
    """Create, build, and train a gensim FastText model.

    Args:
        sentences: A list of tokenized sentences to train on.
        vector_size: The dimensionality of the word vectors.
        min_count: The minimum number of occurrences for included words.
        negative: The number of negative samples.
        window: The maximum distance between current and predicted words.
        cbow: If True, use CBOW; otherwise, use Skip-gram.
        epochs: The number of training iterations.
        seed: The random number generator seed.
        workers: The number of worker threads.

    Returns:
        The trained FastText model.
    """
    model = FastText(
        vector_size=vector_size,
        min_count=min_count,
        negative=negative,
        window=window,
        sg=not cbow,
        seed=seed,
        workers=workers
    )
    model.build_vocab(sentences)
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=epochs
    )

    return model
