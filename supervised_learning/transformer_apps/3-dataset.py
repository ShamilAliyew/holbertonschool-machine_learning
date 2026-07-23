#!/usr/bin/env python3
"""Build a TensorFlow pipeline for a translation dataset."""

import tensorflow as tf
import transformers
from setup import load_pt2en


class Dataset:
    """Load and prepare a dataset for machine translation."""

    def __init__(self, batch_size, max_len):
        """Load, tokenize, filter, and batch the datasets."""
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
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
            .padded_batch(
                batch_size,
                padded_shapes=([None], [None])
            )
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        self.data_valid = (
            self.data_valid
            .filter(length_filter)
            .padded_batch(
                batch_size,
                padded_shapes=([None], [None])
            )
        )

    def tokenize_dataset(self, data):
        """Create Portuguese and English subword tokenizers."""
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        def portuguese_sentences():
            """Yield decoded Portuguese sentences."""
            for pt, _ in data:
                yield pt.numpy().decode('utf-8')

        def english_sentences():
            """Yield decoded English sentences."""
            for _, en in data:
                yield en.numpy().decode('utf-8')

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            portuguese_sentences(),
            vocab_size=2 ** 13
        )
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            english_sentences(),
            vocab_size=2 ** 13
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Encode a Portuguese-English translation pair into tokens."""
        pt_tokens = [self.tokenizer_pt.vocab_size]
        pt_tokens += self.tokenizer_pt.encode(
            pt.numpy().decode('utf-8'),
            add_special_tokens=False
        )
        pt_tokens.append(self.tokenizer_pt.vocab_size + 1)

        en_tokens = [self.tokenizer_en.vocab_size]
        en_tokens += self.tokenizer_en.encode(
            en.numpy().decode('utf-8'),
            add_special_tokens=False
        )
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
