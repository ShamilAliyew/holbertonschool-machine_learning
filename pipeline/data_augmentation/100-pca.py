#!/usr/bin/env python3
"""a function def pca_color(image, alphas):
 that performs PCA color augmentation as described in the AlexNet paper"""
import tensorflow as tf


def pca_color(image, alphas):
    """
    Performs PCA color augmentation as described in the
     AlexNet paper.

    Args:
        image: A 3D tf.Tensor containing the image to change
         (height, width, channels).
               Assumed to be in the range [0, 255] if integer type.
        alphas: A tuple of length 3 containing
        the amount that each channel should change
                (α1, α2, α3 in the paper), typically
                drawn from a Gaussian distribution.

    Returns:
        The augmented image as a tf.Tensor, clipped to
         [0, 255] and cast back to original dtype.
    """
    original_dtype = image.dtype
    image_float = tf.cast(image, tf.float32)

    shape = tf.shape(image_float)
    h, w, c = shape[0], shape[1], shape[2]

    # Reshape image to (num_pixels, channels)
    pixels = tf.reshape(image_float, [-1, c])

    # Calculate the mean of each channel
    mean_pixels = tf.reduce_mean(pixels, axis=0, keepdims=True)

    # Center the pixel data
    centered_pixels = pixels - mean_pixels

    # Calculate the covariance matrix (3x3)
    num_pixels = tf.cast(tf.shape(pixels)[0], tf.float32)
    # Use num_pixels for the denominator as often done for
    # population covariance or when precise unbiased estimate isn't critical for augmentation.
    covariance_matrix = tf.matmul(tf.transpose(centered_pixels), centered_pixels) / num_pixels

    # Perform eigen decomposition
    # eigenvalues will be sorted in ascending order by default for tf.linalg.eigh
    eigenvalues, eigenvectors = tf.linalg.eigh(covariance_matrix)

    # Cast alphas to float32
    alphas = tf.cast(alphas, tf.float32)

    # Calculate the perturbation vector:
    # As per AlexNet paper: sum(p_i * (alpha_i * lambda_i)) for i=1 to 3
    # where p_i are column vectors of eigenvectors
    alpha_eigen_products = alphas * eigenvalues # shape (3,)
    # Multiply eigenvectors by the scalar products (alpha_i * lambda_i)
    perturbation_vector = tf.matmul(eigenvectors, tf.expand_dims(alpha_eigen_products, axis=1))
    perturbation_vector = tf.squeeze(perturbation_vector) # shape (3,)

    # Add the perturbation to all pixels
    augmented_pixels = pixels + perturbation_vector

    # Reshape back to original image shape
    augmented_image = tf.reshape(augmented_pixels, shape)

    # Clip values to [0, 255] and cast back to original dtype
    augmented_image = tf.clip_by_value(augmented_image, 0, 255)
    augmented_image = tf.cast(augmented_image, original_dtype)

    return augmented_image
