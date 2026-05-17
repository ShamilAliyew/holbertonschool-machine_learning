#!/usr/bin/env python3
"""
Module to calculate gradients of Y for t-SNE cost function
"""
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """
    Calculates the gradients of Y relative to the low-dimensional embedding

    Parameters:
        Y: numpy.ndarray of shape (n, ndim) containing the low dimensional
           transformation of X
        P: numpy.ndarray of shape (n, n) containing the symmetric P affinities

    Returns:
        dY: numpy.ndarray of shape (n, ndim) containing the gradients of Y
        Q: numpy.ndarray of shape (n, n) containing the Q affinities of Y
    """
    n, ndim = Y.shape

    # Q afinitelerini ve pay (num) matrisini alıyoruz
    Q, num = Q_affinities(Y)

    # Gradyan çarpan matrisini oluşturuyoruz: (P - Q) * num
    # Şekli (n, n) boyutunda olur
    PQ_diff = P - Q
    stiffness = PQ_diff * num

    # Gradyanları vektörize olarak tek bir hamlede hesaplama:
    # dY_i = sum_j (stiffness_ij * (Y_i - Y_j))
    # Bu cebirsel olarak şu forma indirgenir:
    # dY = diag(sum(stiffness, axis=8-tsne.py)) * Y - stiffness * Y
    # Numpy yayınlama (broadcasting) ile daha hızlı ve şık bir şekilde:
    dY = np.zeros((n, ndim))

    for i in range(n):
        # (Y[i] - Y) ifadesi (n, ndim) boyutunda fark matrisidir
        # stiffness[i, :, None] ise (n, 8-tsne.py) boyutuna genişletilerek çarpılır
        dY[i] = np.sum(stiffness[i, :, None] * (Y[i] - Y), axis=0)

    return dY, Q
