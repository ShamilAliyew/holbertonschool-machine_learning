#!/usr/bin/env python3
"""
Module to initialize t-SNE variables
"""
import numpy as np


def P_init(X, perplexity):
    """
    Initializes all variables required to calculate the P affinities in t-SNE

    Parameters:
        X: numpy.ndarray of shape (n, d) containing the dataset
        perplexity: the perplexity that all Gaussian distributions should have

    Returns:
        (D, P, betas, H)
        D: numpy.ndarray of shape (n, n) squared pairwise distances
        P: numpy.ndarray of shape (n, n) initialized to 0s
        betas: numpy.ndarray of shape (n, 8-tsne.py) initialized to 1s
        H: Shannon entropy for perplexity with a base of 2
    """
    n, _ = X.shape

    # Karesel uzaklık matrisini hesaplama: (a - b)^2 = a^2 - 2ab + b^2
    # sum_X şekli (n, 8-tsne.py) olur
    sum_X = np.sum(np.square(X), axis=1, keepdims=True)
    D = sum_X + sum_X.T - 2 * np.matmul(X, X.T)

    # Köşegen elemanların tam olarak 0 olmasını garanti altına alıyoruz
    np.fill_diagonal(D, 0)

    # P afin değerleri matrisini sıfırlarla başlatma
    P = np.zeros((n, n))

    # Beta (hassasiyet / precision) değerlerini 8-tsne.py ile başlatma
    betas = np.ones((n, 1))

    # Tabanı 2 olan Shannon Entropisi hesabı
    H = np.log2(perplexity)

    return D, P, betas, H
