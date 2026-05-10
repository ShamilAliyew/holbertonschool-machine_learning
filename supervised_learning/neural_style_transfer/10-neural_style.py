#!/usr/bin/env python3
"""Neural Style Transfer"""
import tensorflow as tf
import numpy as np


class NST:
    """Class of NST"""
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1, var=10):
        """Initialize variables"""
        if not isinstance(style_image, np.ndarray) or style_image.ndim != 3 or style_image.shape[2] != 3:
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(content_image, np.ndarray) or content_image.ndim != 3 or content_image.shape[2] != 3:
            raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")
        if not isinstance(var, (int, float)) or var < 0:
            raise TypeError("var must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.var = var
        self.model = self.load_model()
        self.gram_style_features = None
        self.content_feature = None
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """Static Method hat rescales an image
        such that its pixels values are
        between 0 and 1 and its largest side is 512 pixels"""
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")

        h, w, _ = image.shape
        if h > w:
            h_new = 512
            w_new = int(w * 512 / h)
        else:
            w_new = 512
            h_new = int(h * 512 / w)

        image = tf.image.resize(image, (h_new, w_new), method='bicubic')
        image = image / 255.0
        image = tf.expand_dims(image, axis=0)
        return image

    def load_model(self):
        """Load Moodel"""
        vgg = tf.keras.applications.VGG19(include_top=False,
                                          weights='imagenet')
        vgg.trainable = False
        style_outputs = [vgg.get_layer(name).output for name in self.style_layers]
        content_output = vgg.get_layer(self.content_layer).output
        model_outputs = style_outputs + [content_output]
        model = tf.keras.models.Model(inputs=vgg.input, outputs=model_outputs)

        return model

    @staticmethod
    def gram_matrix(input_layer):
        """Gram Martix"""
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)):
            raise TypeError("input_layer must be a tensor of rank 4")
        if len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")
        _, h, w, c = input_layer.shape
        features = tf.reshape(input_layer, (h * w, c))
        gram = tf.matmul(features, features, transpose_a=True)
        gram = gram / tf.cast(h * w, tf.float32)
        gram = tf.expand_dims(gram, axis=0)
        return gram

    def generate_features(self):
        """Generate Features"""
        style_outputs = self.model(self.style_image)
        content_outputs = self.model(self.content_image)

        style_outputs = style_outputs[:len(self.style_layers)]
        content_output = content_outputs[len(self.style_layers):]

        self.gram_style_features = [NST.gram_matrix(style_output) for
                                    style_output in style_outputs]

        self.content_feature = content_output[0]

    def layer_style_cost(self, style_output, gram_target):
        """Layer Style Cost"""
        if not isinstance(style_output, (tf.Tensor, tf.Variable)) or len(style_output.shape) != 4:
            raise TypeError('style_output must be a tensor of rank 4')
        _, h, w, c = style_output.shape
        if not isinstance(gram_target, (tf.Tensor, tf.Variable)) or gram_target.shape != (1, c, c):
            raise TypeError(f'gram_target must be a tensor of shape [1, {c}, {c}]')
        style_output = tf.reshape(style_output, (h * w, c))
        gram_style = tf.matmul(style_output, style_output, transpose_a=True)
        gram_style = tf.expand_dims(gram_style, axis=0)
        loss = tf.reduce_mean(tf.square(gram_style - gram_target)) / (c ** 2)
        return loss

    def style_cost(self, style_outputs):
        """Style Cost"""
        l = len(self.style_layers)

        if not isinstance(style_outputs, list) or len(style_outputs) != l:
            raise TypeError(f'style_outputs must be a list with a length of {l}')

        weight = 1.0 / l
        total_cost = 0

        for i, style_output in enumerate(style_outputs):
            gram_target = self.gram_style_features[i]
            layer_cost = self.layer_style_cost(style_output, gram_target)
            total_cost += weight * layer_cost

        return total_cost

    def content_cost(self, content_output):
        """Content Cost"""
        if not isinstance(content_output, (tf.Tensor, tf.Variable)):
            raise TypeError(f'content_output must be a tensor of shape {self.content_feature.shape}')

        if content_output.shape != self.content_feature.shape:
            raise TypeError(f'content_output must be a tensor of shape {self.content_feature.shape}')

        cost = tf.reduce_mean(tf.square(content_output - self.content_feature))

        return cost

    @staticmethod
    def variational_cost(generated_image):
        """Variational Cost"""
        h_var = tf.reduce_sum(tf.square(
            generated_image[:, 1:, :, :] - generated_image[:, :-1, :, :]
        ))

        v_var = tf.reduce_sum(tf.square(
            generated_image[:, :, 1:, :] - generated_image[:, :, :-1, :]
        ))

        return h_var + v_var

    def total_cost(self, generated_image):
        """Total Cost"""
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)):
            raise TypeError(f'generated_image must be a tensor of shape {self.content_image.shape}')

        if generated_image.shape != self.content_image.shape:
            raise TypeError(f'generated_image must be a tensor of shape {self.content_image.shape}')

        outputs = self.model(generated_image * 255)

        style_outputs = outputs[:len(self.style_layers)]
        content_output = outputs[-1]

        J_content = self.content_cost(content_output)
        J_style = self.style_cost(style_outputs)
        J_var = self.variational_cost(generated_image)

        J_total = self.alpha * J_content + self.beta * J_style + self.var * J_var

        return J_total, J_content, J_style, J_var

    def compute_grads(self, generated_image):
        """Compute Gradients"""
        s = self.content_image.shape
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)) \
                or generated_image.shape != self.content_image.shape:
            raise TypeError(f'generated_image must be a tensor of shape {s}')

        with tf.GradientTape() as tape:
            J_total, J_content, J_style, J_var = self.total_cost(generated_image)
        gradients = tape.gradient(J_total, generated_image)

        return gradients, J_total, J_content, J_style, J_var

    def generate_image(self, iterations=1000,
                       step=None, lr=0.01, beta1=0.9, beta2=0.99):
        """Generated Image"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if step is not None:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step >= iterations:
                raise ValueError(
                    "step must be positive and less than iterations"
                )
        if not isinstance(lr, (float, int)):
            raise TypeError("lr must be a number")
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not isinstance(beta1, float):
            raise TypeError("beta1 must be a float")
        if not (0 <= beta1 <= 1):
            raise ValueError("beta1 must be in the range [0, 1]")
        if not isinstance(beta2, float):
            raise TypeError("beta2 must be a float")
        if not (0 <= beta2 <= 1):
            raise ValueError("beta2 must be in the range [0, 1]")

        generated_image = tf.Variable(self.content_image)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr,
            beta_1=beta1,
            beta_2=beta2
        )

        best_cost = float('inf')
        best_image = None

        for i in range(iterations + 1):
            gradients, J_total, J_content, J_style, J_var = self.compute_grads(
                generated_image
            )
            optimizer.apply_gradients([(gradients, generated_image)])
            generated_image.assign(tf.clip_by_value(generated_image, 0, 1))
            if J_total < best_cost:
                best_cost = J_total
                best_image = generated_image.numpy()

            if step is not None and i % step == 0:
                print(
                    f"Cost at iteration {i}: {J_total.numpy()}, "
                    f"content {J_content.numpy()}, "
                    f"style {J_style.numpy()}, "
                    f"var {J_var.numpy()}"
                )

        best_image = best_image.astype('float32')
        best_image = (best_image - best_image.min()) / (
                best_image.max() - best_image.min()
        )

        return best_image, best_cost
