#!/usr/bin/env python3
"""Write a function def crop_image(image, size):
that performs a random crop of an image"""
import tensorflow as tf


def crop_image(image, size):
    """
    Performs a random crop of an image.

    Args:
        image (tf.Tensor):
         A 3D tf.Tensor containing the image to crop.
        size (tuple):
         A tuple containing the size of the crop
          (height, width, channels).

    Returns:
        tf.Tensor: The cropped image.
    """
    return tf.image.random_crop(image, size=size)
