#!/usr/bin/env python3
"""Tokenize a translation dataset with a TensorFlow wrapper."""

import tensorflow as tf
import transformers
from setup import load_pt2en


class Dataset:
    """Load and prepare a dataset for machine translation."""

    def __init__(self):
        """Load, tokenize, and encode the translation datasets."""
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """Create Portuguese and English subword tokenizers."""
        pt_sentences = []
        en_sentences = []
        for pt, en in data.as_numpy_iterator():
            pt_sentences.append(pt.decode('utf-8'))
            en_sentences.append(en.decode('utf-8'))

        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_sentences,
            vocab_size=2 ** 13
        )
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_sentences,
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
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])

        return pt_tokens, en_tokens
