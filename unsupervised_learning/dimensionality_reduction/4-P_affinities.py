#!/usr/bin/env python3
"""
Module to calculate symmetric P affinities of a dataset using t-SNE
"""
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Calculates the symmetric P affinities of a data set

    Parameters:
        X: numpy.ndarray of shape (n, d) containing the dataset
        tol: maximum tolerance allowed for the difference in Shannon entropy
        perplexity: the perplexity that all Gaussian distributions should have

    Returns:
        P: numpy.ndarray of shape (n, n) containing symmetric P affinities
    """
    D, P, betas, H = P_init(X, perplexity)
    n = X.shape[0]

    for i in range(n):
        # Kendisi hariç diğer noktalara olan mesafeleri alıyoruz
        Di = np.delete(D[i], i)

        # İkili arama (binary search) sınırlarını başlatıyoruz
        beta_min = None
        beta_max = None
        beta = betas[i, 0]

        # İlk entropi ve afinite hesabı
        Hi, Pi = HP(Di, beta)
        H_diff = Hi - H

        # Tolerans sınırları dışındaysa aramaya başla (max 50 iterasyon)
        for _ in range(50):
            if np.abs(H_diff) <= tol:
                break

            # Eğer mevcut entropi hedef entropiden büyükse,
            # varyansı küçültmek (beta'yı büyütmek) gerekir.
            if H_diff > 0:
                beta_min = beta
                if beta_max is None:
                    beta = beta * 2
                else:
                    beta = (beta + beta_max) / 2
            # Eğer mevcut entropi küçükse, beta'yı küçültmek gerekir.
            else:
                beta_max = beta
                if beta_min is None:
                    beta = beta / 2
                else:
                    beta = (beta + beta_min) / 2

            # Yeni beta ile değerleri güncelle
            Hi, Pi = HP(Di, beta)
            H_diff = Hi - H

        # Bulunan beta değerini kaydet
        betas[i, 0] = beta

        # Hesaplanan Pi değerlerini matristeki yerine (kendisi hariç) yerleştir
        P[i, :i] = Pi[:i]
        P[i, i+1:] = Pi[i:]

    # Simetrik P matrisini hesaplama: P_ij = (P_i|j + P_j|i) / (2 * n)
    P = (P + P.T) / (2 * n)

    return P
