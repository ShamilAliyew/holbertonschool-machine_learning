#!/usr/bin/env python3
"""a function def cat_matrices2D(mat1, mat2, axis=0):
 that concatenates two matrices along a specific axis"""
shape = __import__('2-size_me_please').matrix_shape


def cat_matrices2D(mat1, mat2, axis=0):
    """a function def cat_matrices2D(mat1, mat2, axis=0):
     that concatenates two matrices along a specific axis"""
    if not mat1 or not mat2:
        return None

    if axis == 0:
        if len (mat1[0]) != len(mat2[0]):
            return None
        result = []
        for row in mat1:
            result.append(row[:])
        for row in mat2:
            result.append(row[:])
        return result

    if axis == 1:
        if (len(mat1) != len(mat2)):
            return None

        cat_metrice = []
        for i in range(len(mat1)):
            cat_metrice.append(mat1[i] + mat2[i])
        return cat_metrice
    return None

