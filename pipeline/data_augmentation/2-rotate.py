#!/usr/bin/env python3
"""a function def rotate_image(image):
 that rotates an image by 90 degrees counter-clockwise:"""
import tensorflow as tf


def rotate_image(image):
    """
    Rotates an image by 90 degrees counter-clockwise.

    Args:
        image (tf.Tensor):
         A 3D tf.Tensor containing the image to rotate.

    Returns:
        tf.Tensor: The rotated image.
    """
    return tf.image.rot90(image, k=1)
