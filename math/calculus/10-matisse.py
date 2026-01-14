#!/usr/bin/env python3
"""a function that calculates the derivative of a polynomial"""


def poly_derivative(poly):
    """a function that calculates the derivative of a polynomial"""
    if (not isinstance(poly, list) or len(poly) == 0
            or any(not isinstance(x, (int, float)) for x in poly)):
        return None
    if len(poly) == 1:
        return [0]
    derivative = []
    for i in range(1, len(poly)):
        cof = poly[i] * i
        derivative.append(cof)
    return derivative
