#!/usr/bin/env python3
"""Documented"""
import numpy as np


def specificity(confusion):
    """Documented
    TN / TN + FP
    """
    TP = np.diagonal(confusion)
    FP = np.sum(confusion, axis=0) - TP
    FN = np.sum(confusion, axis=1) - TP
    TN = np.sum(confusion) - (TP+FP+FN)
    return TN / (TN+FP)
