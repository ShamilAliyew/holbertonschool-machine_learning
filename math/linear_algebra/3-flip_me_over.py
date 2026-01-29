#!/usr/bin/env python3
"""a function def matrix_transpose(matrix):
that returns the transpose of a 2D matrix, matrix"""


def matrix_transpose(matrix):
    """a function def matrix_transpose(matrix):
    that returns the transpose of a 2D matrix, matrix"""
    transposed_matrix = []
    # for j in range(len(matrix[0])):
    #     transposed_row = []
    #     for i in range(len(matrix)):
    #         transposed_row.append(matrix[i][j])
    #     transposed_matrix.append(transposed_row)
    # return transposed_matrix
    for row in zip(*matrix):
        transposed_matrix.append(list(row))
    return transposed_matrix
