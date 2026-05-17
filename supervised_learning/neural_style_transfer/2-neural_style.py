#!/usr/bin/env python3
"""
Module to perform Neural Style Transfer with Gram Matrix calculation
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
        self.load_model()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixels values are between 0 and 1
        and its largest side is 512 pixels
        """
        if not isinstance(image, np.ndarray) or \
           len(image.shape) != 3 or image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, _ = image.shape

        if h >= w:
            h_new = 512
            w_new = int(round(w * (512 / h)))
        else:
            w_new = 512
            h_new = int(round(h * (512 / w)))

        image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        image_tensor = tf.expand_dims(image_tensor, axis=0)

        scaled_image = tf.image.resize(
            image_tensor,
            size=[h_new, w_new],
            method=tf.image.ResizeMethod.BICUBIC
        )

        scaled_image = scaled_image / 255.0
        scaled_image = tf.clip_by_value(scaled_image, 0.0, 1.0)

        return scaled_image

    def load_model(self):
        """
        Creates the model used to calculate cost using VGG19 Keras model
        """
        vgg = tf.keras.applications.vgg19.VGG19(
            include_top=False,
            weights='imagenet'
        )
        vgg.trainable = False

        outputs = []
        for name in self.style_layers:
            outputs.append(vgg.get_layer(name).output)
        outputs.append(vgg.get_layer(self.content_layer).output)

        custom_model = tf.keras.models.Model(vgg.input, outputs)

        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        config = custom_model.get_config()

        for layer in config['layers']:
            if layer['class_name'] == 'MaxPooling2D':
                layer['class_name'] = 'AveragePooling2D'
                if 'config' in layer:
                    layer['config']['name'] = layer['config']['name'].replace(
                        'pool', 'pool'
                    )

        self.model = tf.keras.models.Model.from_config(
            config,
            custom_objects=custom_objects
        )
        self.model.set_weights(custom_model.get_weights())

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculates the gram matrix of a layer output

        Parameters:
            input_layer: tf.Tensor or tf.Variable of
            shape (1, h, w, c)

        Returns:
            gram: tf.Tensor of shape (1, c, c)
             containing the gram matrix
        """
        # Tensor və ya Variable olmasını, həmçinin
        # rank-ın 4 olmasını yoxlayırıq
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or \
           len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        # Ölçüləri götürürük (batch həmişə 1-dir)
        _, h, w, c = input_layer.shape

        # Hündürlük və eni vahid bir məkana düzləşdiririk (h * w)
        # Nəticə forması: (h * w, c)
        features = tf.reshape(input_layer, [-1, c])

        # Gram matrisini qururuq: F^T * F (Ölçü: c, c)
        gram = tf.matmul(features, features, transpose_a=True)

        # Matrisi cəmi piksel sayına bölərək normallaşdırırıq (h * w)
        num_locations = tf.cast(h * w, tf.float32)
        gram = gram / num_locations

        # Tələb olunan çıxış formasına
        # (1, c, c) uyğun olaraq batch ölçüsü əlavə edirik
        gram = tf.expand_dims(gram, axis=0)

        return gram
