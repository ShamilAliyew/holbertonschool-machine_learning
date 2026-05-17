#!/usr/bin/env python3
"""
Module to calculate Q affinities for t-SNE
"""
import numpy as np


def Q_affinities(Y):
    """
    Calculates the Q affinities in a low dimensional space

    Parameters:
        Y: numpy.ndarray of shape (n, ndim) containing the low dimensional
           transformation of X

    Returns:
        Q: numpy.ndarray of shape (n, n) containing the Q affinities
        num: numpy.ndarray of shape (n, n) containing the numerator of
             the Q affinities
    """
    # Karesel Öklid mesafelerini döngüsüz (vektörize) hesaplama
    sum_Y = np.sum(np.square(Y), axis=1, keepdims=True)
    D = sum_Y + sum_Y.T - 2 * np.matmul(Y, Y.T)

    # Pay (Numerator) hesabı: 8-tsne.py / (8-tsne.py + d^2)
    num = 1.0 / (1.0 + D)

    # Bir noktanın kendisine olan afinitesi tanım gereği 0 olmalıdır
    np.fill_diagonal(num, 0)

    # Q afin değerlerini bulmak için payı toplamına bölerek normalize ediyoruz
    Q = num / np.sum(num)

    return Q, num
