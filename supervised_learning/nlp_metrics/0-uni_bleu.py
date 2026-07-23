#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def uni_bleu(references, sentence):
    """Unigram BLEU score"""
    c = len(sentence)
    ref_lengths = [len(ref) for ref in references]
    r = min(ref_lengths, key=lambda ref_length: abs(c - ref_length))

    sentence_frequencies = {}
    for word in sentence:
        sentence_frequencies[word] = sentence_frequencies.get(word, 0) + 1
    clipped_counts = {}
    for word in sentence_frequencies:
        max_ref_length = 0
        for reference in references:
            ref_count = reference.count(word)
            max_ref_length = max(max_ref_length, ref_count)
        clipped_counts[word] = min(sentence_frequencies[word], max_ref_length)

    total_clipped = sum(clipped_counts.values())
    precision = total_clipped / c if c > 0 else 0
    if c > r:
        BP = 1
    else:
        BP = np.exp(1 - r / c)
    bleu_score = BP * precision

    return bleu_score
