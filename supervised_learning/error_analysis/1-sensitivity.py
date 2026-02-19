#!/usr/bin/env python3
"""Documented"""
import numpy as np


def sensitivity(confusion):
    """Documented"""
    TP = np.diagonal(confusion)
    P = np.sum(confusion, axis=1)
    return TP / P
