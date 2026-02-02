#!/usr/bin/env python3
"""a function def inverse(matrix):
 that calculates the inverse of a matrix:"""
adjugate = __import__('3-adjugate').adjugate
determinant = __import__('0-determinant').determinant


def inverse(matrix):
    """a function def inverse(matrix):
     that calculates the inverse of a matrix:"""
    adjugate_matrix = adjugate(matrix)
    determinant_of_matrix = determinant(matrix)
    if determinant_of_matrix == 0:
        return None
    inverse_matrix = [[0 for _ in range(len(matrix))]
                      for _ in range(len(matrix))]
    for i in range(len(adjugate_matrix)):
        for j in range(len(adjugate_matrix)):
            inverse_matrix[i][j] =\
                adjugate_matrix[i][j] / determinant_of_matrix
    return inverse_matrix
