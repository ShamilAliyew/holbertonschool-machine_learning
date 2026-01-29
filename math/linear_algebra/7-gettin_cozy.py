#!/usr/bin/env python3
"""a function def cat_matrices2D(mat1, mat2, axis=0):
 that concatenates two matrices along a specific axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """a function def cat_matrices2D(mat1, mat2, axis=0):
     that concatenates two matrices along a specific axis"""
    if axis == 0:
        return mat1 + mat2
    elif axis == 1:
        cat_metrice = []
        for i in range(len(mat1)):
            cat_metrice.append(mat1[i] + mat2[i])
        return cat_metrice
