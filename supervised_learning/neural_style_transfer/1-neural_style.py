#!/usr/bin/env python3
"""
Module to perform Neural Style Transfer model loading
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
        # Modeli yükləyirik və instance attribute olaraq saxlayırıq
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
        Creates the model used to calculate cost using VGG19 Keras model.
        Replaces MaxPooling2D layers with AveragePooling2D layers.
        Outputs are listed in style_layers followed by content_layer.
        """
        # 1. Hazır VGG19 modelini ImageNet çəkiləri ilə yükləyirik
        vgg = tf.keras.applications.vgg19.VGG19(
            include_top=False,
            weights='imagenet'
        )
        # Çəkilərin dəyişməməsi üçün təlimi dondururuq
        vgg.trainable = False

        # 2. Bizə lazım olan çıxış laylarının siyahısını hazırlayırıq
        outputs = []
        for name in self.style_layers:
            outputs.append(vgg.get_layer(name).output)
        outputs.append(vgg.get_layer(self.content_layer).output)

        # 3. Giriş qatını saxlayaraq xüsusi alt modelimizi qururuq
        custom_model = tf.keras.models.Model(vgg.input, outputs)

        # 4. MaxPooling laylarını AveragePooling2D ilə əvəzləyirik
        # Bu, model_summary strukturunun əsas test
        # nümunəsinə uyğun olması üçündür
        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}

        # Modeli konfiqurasiya səviyyəsində yenidən qurmaq üçün clone edirik
        config = custom_model.get_config()

        # Konfiqurasiyadakı MaxPooling2D təbəqələrini AveragePooling2D edirik
        for layer in config['layers']:
            if layer['class_name'] == 'MaxPooling2D':
                layer['class_name'] = 'AveragePooling2D'
                # Default Hovuz parametrlərini qoruyuruq
                if 'config' in layer:
                    layer['config']['name'] = layer['config']['name'].replace(
                        'pool', 'pool'
                    )

        # Yenilənmiş konfiqurasiyadan yeni modeli
        # yaradırıq və çəkiləri köçürürük
        self.model = tf.keras.models.Model.from_config(
            config,
            custom_objects=custom_objects
        )
        self.model.set_weights(custom_model.get_weights())
