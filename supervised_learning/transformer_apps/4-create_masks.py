#!/usr/bin/env python3
"""Create attention masks for Transformer training."""

import tensorflow as tf


def create_masks(inputs, target):
    """Create encoder, combined, and decoder attention masks."""
    input_padding = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    input_padding = input_padding[:, tf.newaxis, tf.newaxis, :]

    target_padding = tf.cast(tf.math.equal(target, 0), tf.float32)
    target_padding = target_padding[:, tf.newaxis, tf.newaxis, :]

    target_length = tf.shape(target)[1]
    look_ahead = 1 - tf.linalg.band_part(
        tf.ones((target_length, target_length)),
        -1,
        0
    )
    combined_mask = tf.maximum(target_padding, look_ahead)

    encoder_mask = input_padding
    decoder_mask = input_padding

    return encoder_mask, combined_mask, decoder_mask
