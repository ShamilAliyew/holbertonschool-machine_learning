#!/usr/bin/env python3
"""calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """calculates the integral of a polynomial"""
    if not isinstance(poly, list) or any(not isinstance(x, (int, float)) for x in poly)\
        or not isinstance(C, int):
        return None
    result= [C]
    for i in range(len(poly)):
        coefficient = poly[i] / (i+1)
    return result
