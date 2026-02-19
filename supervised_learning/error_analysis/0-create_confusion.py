#!/usr/bin/env python3
"""Documented"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """Documented"""
    return np.matmul(labels.T, logits)
