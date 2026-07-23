#!/usr/bin/env python3
"""Load and prepare a dataset for machine translation."""

import transformers
from setup import load_pt2en


class Dataset:
    """Load Portuguese-to-English data and create its tokenizers."""

    def __init__(self):
        """Load the training and validation datasets."""
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
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
