#!/usr/bin/env python3
"""Documented"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """Documented
    2 * prec * sens / prec + sens
    """
    return (2 * precision(confusion) * sensitivity(confusion) /
            (precision(confusion) + sensitivity(confusion)))
