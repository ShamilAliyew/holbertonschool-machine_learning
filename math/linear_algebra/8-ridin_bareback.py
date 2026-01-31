#!/usr/bin/env python3
"""a function def mat_mul(mat1, mat2):
 that performs matrix multiplication"""


def mat_mul(mat1, mat2):
    """a function def mat_mul(mat1, mat2):
     that performs matrix multiplication"""
    if not mat1 or not mat2:
        return None

    if not mat1[0] or not mat2[0]:
        return None

    if len(mat1[0]) != len(mat2):
        return None
    mat2_transpose = list(zip(*mat2))
    result = []
    for row in mat1:
        result.append([
            sum(a*b for a, b in zip(row, col))
            for col in mat2_transpose])

    return result
