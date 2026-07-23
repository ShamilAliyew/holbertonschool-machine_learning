#!/usr/bin/env python3
"""Build a TensorFlow pipeline for a translation dataset."""

import tensorflow as tf
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """Load and prepare a dataset for machine translation."""

    def __init__(self, batch_size, max_len):
        """Load, tokenize, filter, and batch the datasets."""
        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='train',
            as_supervised=True
        )
        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='validation',
            as_supervised=True
        )
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

        def length_filter(pt, en):
            """Return whether both sentences fit within max_len."""
            return tf.logical_and(
                tf.size(pt) <= max_len,
                tf.size(en) <= max_len
            )

        self.data_train = (
            self.data_train
            .filter(length_filter)
            .cache()
            .shuffle(20000)
            .padded_batch(batch_size)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        self.data_valid = (
            self.data_valid
            .filter(length_filter)
            .padded_batch(batch_size)
        )

    def tokenize_dataset(self, data):
        """Create Portuguese and English subword tokenizers."""
        tokenizer_pt = (
            tfds.features.text.SubwordTextEncoder.build_from_corpus(
                (pt.numpy() for pt, en in data),
                target_vocab_size=2 ** 15
            )
        )
        tokenizer_en = (
            tfds.features.text.SubwordTextEncoder.build_from_corpus(
                (en.numpy() for pt, en in data),
                target_vocab_size=2 ** 15
            )
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Encode a Portuguese-English translation pair into tokens."""
        pt_tokens = [self.tokenizer_pt.vocab_size]
        pt_tokens += self.tokenizer_pt.encode(pt.numpy())
        pt_tokens.append(self.tokenizer_pt.vocab_size + 1)

        en_tokens = [self.tokenizer_en.vocab_size]
        en_tokens += self.tokenizer_en.encode(en.numpy())
        en_tokens.append(self.tokenizer_en.vocab_size + 1)

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """Wrap encode for use in a TensorFlow data pipeline."""
        pt_tokens, en_tokens = tf.py_function(
            self.encode,
            [pt, en],
            [tf.int64, tf.int64]
        )
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])

        return pt_tokens, en_tokens
