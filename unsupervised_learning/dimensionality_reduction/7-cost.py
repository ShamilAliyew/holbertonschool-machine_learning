#!/usr/bin/env python3
"""
Module to calculate the cost of t-SNE transformation
"""
import numpy as np


def cost(P, Q):
    """
    Calculates the cost of the t-SNE transformation

    Parameters:
        P: numpy.ndarray of shape (n, n) containing the P affinities
        Q: numpy.ndarray of shape (n, n) containing the Q affinities

    Returns:
        C: the cost of the transformation (KL divergence)
    """
    # log(0) hatalarından kaçınmak için p ve
    # q matrislerini alt sınıra eşitleriz
    P = np.maximum(P, 1e-12)
    Q = np.maximum(Q, 1e-12)

    # KL Iraksaması hesabı
    C = np.sum(P * np.log(P / Q))

    return C
