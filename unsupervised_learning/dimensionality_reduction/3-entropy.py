#!/usr/bin/env python3
"""
Module to calculate Shannon entropy and P affinities relative to a data point
"""
import numpy as np


def HP(Di, beta):
    """
    Calculates the Shannon entropy and P affinities relative to a data point

    Parameters:
        Di: numpy.ndarray of shape (n - 1,) containing pairwise distances
        beta: numpy.ndarray of shape (1,) containing the beta value

    Returns:
        (Hi, Pi)
        Hi: the Shannon entropy of the points
        Pi: numpy.ndarray of shape (n - 1,) containing the P affinities
    """
    # Eksponent payını hesapla: exp(-beta * Di)
    # beta[0] kullanılarak skaler bir değerle çarpım sağlanır
    distances_conditioned = -Di * beta[0]
    exp_distances = np.exp(distances_conditioned)

    # Toplam exp değerini hesapla (Payda)
    sum_exp = np.sum(exp_distances)

    # Afinite olasılıklarını normalize et (Pi)
    Pi = exp_distances / sum_exp

    # Shannon Entropisi hesabı (H_i = -sum(P * log2(P)))
    # log2(0) hatasından (NaN) kaçınmak için çok küçük bir pay (1e-7) eklenebilir
    # Veya matematiksel basitleştirme formülü doğrudan kullanılabilir:
    Hi = (beta[0] * np.sum(Pi * Di) / np.log(2)) + np.log2(sum_exp)

    return Hi, Pi
