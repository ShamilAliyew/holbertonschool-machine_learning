#!/usr/bin/env python3
"""
Neural Style Transfer class with gradient computation
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
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image to 512 pixels max side
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
        Creates the model using VGG19
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
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or \
           len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        _, h, w, c = input_layer.shape

        features = tf.reshape(input_layer, [-1, c])
        gram = tf.matmul(features, features, transpose_a=True)

        num_locations = tf.cast(h * w, tf.float32)
        gram = gram / num_locations
        gram = tf.expand_dims(gram, axis=0)

        return gram

    def generate_features(self):
        """
        Extracts the features for style and content
        """
        style_p = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255.0
        )
        content_p = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255.0
        )

        style_outputs = self.model(style_p)
        self.gram_style_features = [
            self.gram_matrix(layer) for layer in style_outputs[:-1]
        ]

        content_outputs = self.model(content_p)
        self.content_feature = content_outputs[-1]

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculates the style cost for a single layer
        """
        if not isinstance(style_output, (tf.Tensor, tf.Variable)) or \
           len(style_output.shape) != 4:
            raise TypeError("style_output must be a tensor of rank 4")

        c = style_output.shape[-1]

        if not isinstance(gram_target, (tf.Tensor, tf.Variable)) or \
           gram_target.shape != (1, c, c):
            raise TypeError(
                "gram_target must be a tensor of shape [1, {0}, {0}]".format(c)
            )

        gram_style = self.gram_matrix(style_output)
        cost = tf.reduce_mean(tf.square(gram_style - gram_target))

        return cost

    def style_cost(self, style_outputs):
        """
        Calculates the style cost for the generated image
        """
        l_layers = len(self.style_layers)

        if not isinstance(style_outputs, list) or \
           len(style_outputs) != l_layers:
            raise TypeError(
                "style_outputs must be a list with a length of {}"
                .format(l_layers)
            )

        total_style_cost = 0.0
        for i in range(l_layers):
            total_style_cost += self.layer_style_cost(
                style_outputs[i],
                self.gram_style_features[i]
            )

        total_style_cost /= l_layers

        return total_style_cost

    def content_cost(self, content_output):
        """
        Calculates the content cost for the generated image
        """
        s_shape = self.content_feature.shape

        if not isinstance(content_output, (tf.Tensor, tf.Variable)) or \
           content_output.shape != s_shape:
            raise TypeError(
                "content_output must be a tensor of shape {}"
                .format(s_shape)
            )

        cost = tf.reduce_mean(tf.square(content_output - self.content_feature))

        return cost

    def total_cost(self, generated_image):
        """
        Calculates the total cost for the generated image
        """
        c_shape = self.content_image.shape

        if not isinstance(generated_image, (tf.Tensor, tf.Variable)) or \
           generated_image.shape != c_shape:
            raise TypeError(
                "generated_image must be a tensor of shape {}"
                .format(c_shape)
            )

        preprocessed = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255.0
        )

        outputs = self.model(preprocessed)
        style_outputs = outputs[:-1]
        content_output = outputs[-1]

        J_content = self.content_cost(content_output)
        J_style = self.style_cost(style_outputs)

        J = (self.alpha * J_content) + (self.beta * J_style)

        return (J, J_content, J_style)

    def compute_grads(self, generated_image):
        """
        Calculates the gradients for the generated image

        Parameters:
            generated_image: tf.Tensor/tf.Variable of shape (1, nh, nw, 3)

        Returns:
            gradients, J_total, J_content, J_style
        """
        # Hədəf struktur formasını götürürük
        c_shape = self.content_image.shape

        # Girişin növ və forma doğrulaması
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)) or \
           generated_image.shape != c_shape:
            raise TypeError(
                "generated_image must be a tensor of shape {}"
                .format(c_shape)
            )

        # Qradiyent tape yaradılır
        with tf.GradientTape() as tape:
            # Əgər giriş tf.Tensor olarsa, onun izlənməsini təmin edirik
            if isinstance(generated_image, tf.Tensor):
                tape.watch(generated_image)

            # Cari şəklin ümumi və fərdi itkiləri hesablanır
            J_total, J_content, J_style = self.total_cost(generated_image)

        # Ümumi itkinin (J_total) şəklə nəzərən qradiyentləri hesablanır
        gradients = tape.gradient(J_total, generated_image)

        return gradients, J_total, J_content, J_style
