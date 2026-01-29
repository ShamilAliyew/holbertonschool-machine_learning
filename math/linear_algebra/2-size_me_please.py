#!/usr/bin/env python3
"""a function def matrix_shape(matrix):
 that calculates the shape of a matrix"""



def matrix_shape(matrix):
    """a function def matrix_shape(matrix):
     that calculates the shape of a matrix"""
    matrix_shape = []
    while isinstance(matrix ,list):
        matrix_shape.append(len(matrix))
        if len(matrix)==0:
            break
        matrix = matrix[0]
    return matrix_shape
