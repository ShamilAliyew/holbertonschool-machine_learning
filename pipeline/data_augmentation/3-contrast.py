#!/usr/bin/env python3
"""a function def change_contrast(image, lower, upper):
 that randomly adjusts the contrast of an image.
"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Randomly adjusts the contrast of an image.

    Args:
        image (tf.Tensor): A 3D tf.Tensor representing the input image to adjust the contrast.
        lower (float): A float representing the lower bound of the random contrast factor range.
        upper (float): A float representing the upper bound of the random contrast factor range.

    Returns:
        tf.Tensor: The contrast-adjusted image.
    """
    return tf.image.random_contrast(image, lower=lower, upper=upper)
