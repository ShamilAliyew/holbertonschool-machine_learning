#!/usr/bin/env python3
"""
Module to perform Neural Style Transfer initialization
"""
import numpy as np
import tensorflow as tf


class NST:
    """
    NST class to perform tasks for neural style transfer
    """
    style_layers = [
        'block1_conv1', 'block2_conv1',
        'block3_conv1', 'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor for NST

        Parameters:
            style_image: numpy.ndarray, style reference image
            content_image: numpy.ndarray, content reference image
            alpha: float, weight for content cost
            beta: float, weight for style cost
        """
        if not isinstance(style_image, np.ndarray) or \
           len(style_image.shape) != 3 or style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(content_image, np.ndarray) or \
           len(content_image.shape) != 3 or content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixels values are between 0 and 1
        and its largest side is 512 pixels

        Parameters:
            image: numpy.ndarray of shape (h, w, 3) to be scaled

        Returns:
            scaled_image: tf.Tensor of shape (1, h_new, w_new, 3)
        """
        if not isinstance(image, np.ndarray) or \
           len(image.shape) != 3 or image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, _ = image.shape

        # Ən böyük tərəfi tapıb nisbəti qoruyaraq yeni ölçüləri təyin edirik
        if h >= w:
            h_new = 512
            w_new = int(round(w * (512 / h)))
        else:
            w_new = 512
            h_new = int(round(h * (512 / w)))

        # Şəkli tf.Tensor-a çeviririk və batch ölçüsü (1) əlavə edirik
        image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        image_tensor = tf.expand_dims(image_tensor, axis=0)

        # Bicubic interpolasiya ilə ölçüləndirmə (resize)
        scaled_image = tf.image.resize(
            image_tensor,
            size=[h_new, w_new],
            method=tf.image.ResizeMethod.BICUBIC
        )

        # Pikselləri [0, 255] aralığından [0, 1] aralığına normallaşdırırıq
        scaled_image = scaled_image / 255.0

        # Dəyərlərin tam [0, 1] aralığında
        # qalmasını təmin etmək üçün clip edirik
        scaled_image = tf.clip_by_value(scaled_image, 0.0, 1.0)

        return scaled_image
