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
        beta: numpy.ndarray or float containing the beta value

    Returns:
        (Hi, Pi)
        Hi: the Shannon entropy of the points
        Pi: numpy.ndarray of shape (n - 1,) containing the P affinities
    """
    # beta hem array (1,) hem de float gelebileceği için güvenli dönüşüm yapıyoruz
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

    # Shannon Entropisi hesabı (H_i)
    # Taşma ve NaN hatalarını önleyen basitleştirilmiş formül
    Hi = (beta_val * np.sum(Pi * Di) / np.log(2)) + np.log2(sum_exp)

    return Hi, Pi
