#!/usr/bin/env python3
"""Return the integer value of the sum"""


def summation_i_squared(n):
    """Return the integer value of the sum"""
    if isinstance(n, int) and n > 0:
        return int((n*(n+1)*(2*n+1))/6)
    else:
        return None
