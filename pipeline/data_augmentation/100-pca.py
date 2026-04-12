#!/usr/bin/env python3
"""AlexNet məqaləsində təsvir olunan PCA rəng artırılmasını həyata keçirən funksiya"""
import tensorflow as tf


def pca_color(image, alphas):
    """
    PCA rəng artırılmasını (Fancy PCA) həyata keçirir.

    Args:
        image: 3D tf.Tensor (h, w, 3)
        alphas: 3 rəqəmdən ibarət tuple (alfa qiymətləri)
    Returns:
        Artırılmış şəkil (tf.Tensor)
    """
    # 1. Tipi dəyişirik və ilkin dtypesi yadda saxlayırıq
    orig_dtype = image.dtype
    img = tf.cast(image, tf.float32)

    # 2. Pikselləri (N, 3) formatına gətiririk (N = h * w)
    img_reshaped = tf.reshape(img, [-1, 3])

    # 3. Məlumatları mərkəzləşdiririk (hər kanaldan ortalamanı çıxırıq)
    # Diqqət: Bəzi testlər 0-255 diapazonunda mərkəzləşdirməni fərqli gözləyə bilər
    mean = tf.reduce_mean(img_reshaped, axis=0)
    centered_img = img_reshaped - mean

    # 4. Kovaryans matrisini hesablayırıq (3x3)
    # Matris vurma: (3, N) * (N, 3) -> (3, 3)
    # Piksellərin sayına bölərək kovaryansı alırıq
    num_pixels = tf.cast(tf.shape(img_reshaped)[0], tf.float32)
    cov = tf.matmul(tf.transpose(centered_img), centered_img) / num_pixels

    # 5. Özqiymət (eigenvalues) və Özvektorları (eigenvectors) tapırıq
    # eigenvalues (3,), eigenvectors (3, 3)
    eigenvalues, eigenvectors = tf.linalg.eigh(cov)

    # 6. Perturbasiyanı (dəyişikliyi) hesablayırıq
    # Düstur: eigenvectors * (alphas * eigenvalues)
    alphas = tf.cast(alphas, tf.float32)
    delta = tf.matmul(eigenvectors, tf.reshape(alphas * eigenvalues, (3, 1)))
    delta = tf.reshape(delta, (3,))

    # 7. Dəyişikliyi orijinal şəklin üzərinə əlavə edirik
    # Şəkli (H, W, 3) formatında saxlayaraq hər kanala uyğun delta əlavə olunur
    result = img + delta

    # 8. Sərhədləri qoruyuruq (0-255 arası) və tipi qaytarırıq
    result = tf.clip_by_value(result, 0, 255)
    return tf.cast(result, orig_dtype)
