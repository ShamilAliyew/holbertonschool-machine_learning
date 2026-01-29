#!/usr/bin/env python3
"""a function def add_matrices2D(mat1, mat2):
 that adds two matrices element-wise"""
shape = __import__('2-size_me_please').matrix_shape


def add_matrices2D(mat1, mat2):
    """a function def add_matrices2D(mat1, mat2):
     that adds two matrices element-wise"""
    sum_metrice = []
    if shape(mat1) != shape(mat2):
        return None
    else:
        for i in range(len(mat1)):
            temp = []
            for j in range(len(mat1[i])):
                temp.append(mat1[i][j] + mat2[i][j])
            sum_metrice.append(temp)
        return sum_metrice
