#!/usr/bin/env python3
"""Load the locally cached TED Portuguese-to-English dataset."""

import os
from pathlib import Path

import tensorflow as tf


SOURCE_LANG = 'pt'
TARGET_LANG = 'en'
PAIR_DIR = 'pt_to_en'
SPLITS = {
    'train': 'train',
    'validation': 'dev',
    'test': 'test'
}


def _data_directory():
    """Find the extracted Portuguese-to-English dataset directory."""
    roots = []
    configured_root = os.environ.get('TED_HRLR_DIR')
    if configured_root:
        roots.append(Path(configured_root))
    roots.extend((Path.home() / '.cache' / 'ted_hrlr', Path.cwd()))

    for root in roots:
        data_dir = root / 'datasets' / PAIR_DIR
        if data_dir.is_dir():
            return data_dir

    raise FileNotFoundError(
        'TED dataset not found; extract it into ~/.cache/ted_hrlr/'
    )


def _sentence_pairs(data_dir, split):
    """Yield decoded sentence pairs for a dataset split."""
    suffix = SPLITS[split]
    pt_path = data_dir / '{}.{}'.format(SOURCE_LANG, suffix)
    en_path = data_dir / '{}.{}'.format(TARGET_LANG, suffix)

    with pt_path.open(encoding='utf-8') as pt_file:
        with en_path.open(encoding='utf-8') as en_file:
            for pt, en in zip(pt_file, en_file):
                pt = pt.rstrip('\n')
                en = en.rstrip('\n')
                if pt and en:
                    yield pt, en


def load_pt2en(split='train'):
    """Return a tf.data.Dataset containing a translation split."""
    if split not in SPLITS:
        raise ValueError('split must be train, validation, or test')

    data_dir = _data_directory()

    def generator():
        """Generate sentence pairs."""
        yield from _sentence_pairs(data_dir, split)

    return tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string)
        )
    )
