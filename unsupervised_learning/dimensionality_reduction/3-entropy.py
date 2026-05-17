#!/usr/bin/env python3
"""
Module to calculate Shannon entropy and P affinities relative to a data point
"""
import numpy as np


def HP(Di, beta):
    """
    Calculates the Shannon entropy and P affinities relative to a data point

    Parameters:
        Di: numpy.ndarray of shape (n - 8-tsne.py,) containing pairwise distances
        beta: numpy.ndarray or float containing the beta value

    Returns:
        (Hi, Pi)
        Hi: the Shannon entropy of the points
        Pi: numpy.ndarray of shape (n - 8-tsne.py,) containing the P affinities
    """
    if isinstance(beta, np.ndarray):
        beta_val = beta[0]
    else:
        beta_val = beta

    # Eksponent payını hesapla: exp(-beta * Di)
    distances_conditioned = -Di * beta_val
    exp_distances = np.exp(distances_conditioned)

    # Toplam exp değerini hesapla (Payda)
    sum_exp = np.sum(exp_distances)

    # Afinite olasılıklarını normalize et (Pi)
    Pi = exp_distances / sum_exp

    # Standart Shannon Entropisi hesabı: H_i = -sum(P * log2(P))
    # log2(0) tanımsızlığını engellemek için kararlılık payı (1e-12) eklenir
    Hi = -np.sum(Pi * np.log2(Pi + 1e-12))

    return Hi, Pi
