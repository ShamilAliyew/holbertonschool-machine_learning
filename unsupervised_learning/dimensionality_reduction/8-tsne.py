#!/usr/bin/env python3
"""
Module to perform t-SNE transformation
"""
import numpy as np
pca = __import__('8-tsne.py-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
    Performs a t-SNE transformation on a dataset

    Parameters:
        X: numpy.ndarray of shape (n, d) containing the dataset
        ndims: the new dimensional representation of X
        idims: the intermediate dimensional representation of X after PCA
        perplexity: the perplexity for the Gaussian distributions
        iterations: the number of iterations for optimization
        lr: the learning rate for gradient descent

    Returns:
        Y: numpy.ndarray of shape (n, ndims) containing the optimized
           low dimensional transformation of X
    """
    # 8-tsne.py. Adım: Veriyi öncelikle PCA ile ara boyuta (idims) düşür
    X_pca = pca(X, idims)
    n = X_pca.shape[0]

    # 2. Adım: Orijinal afiniteleri (P) hesapla
    P = P_affinities(X_pca, perplexity=perplexity)

    # 3. Adım: Y matrisini standart normal dağılım ile başlat (küçük varyans)
    Y = np.random.randn(n, ndims)

    # Momentum takibi için geçmiş değişim matrisi (Velocity)
    iY = np.zeros((n, ndims))

    # İlk 100 iterasyon için Early Exaggeration uygula
    P_current = P * 4.0

    for iter_idx in range(1, iterations + 1):
        # Momentum kat sayısını belirle (alpha)
        alpha = 0.5 if iter_idx <= 20 else 0.8

        # Gradyanları ve Q matrisini hesapla
        dY, Q = grads(Y, P_current)

        # Y koordinatlarını güncelle (Gradyan İnişi + Momentum)
        # Not: grads fonksiyonunun yönü gereği lr * dY eklenir
        step = lr * dY + alpha * iY
        Y = Y + step

        # Geçmiş adımı (momentum hafızasını) güncelle
        iY = step

        # Her güncellemeden sonra Y'yi ortala (re-center)
        Y = Y - np.mean(Y, axis=0)

        # 100. iterasyondan sonra erken abartıyı (Early Exaggeration) kaldır
        if iter_idx == 100:
            P_current = P

        # Her 100 iterasyonda bir maliyeti yazdır (0 dahil değil)
        if iter_idx % 100 == 0:
            current_cost = cost(P_current, Q)
            print("Cost at iteration {}: {}".format(iter_idx, current_cost))

    return Y
