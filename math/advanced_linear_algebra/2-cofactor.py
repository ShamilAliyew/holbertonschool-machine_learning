#!/usr/bin/env python3
"""a function def cofactor(matrix):
 that calculates the cofactor matrix of a matrix"""
minor = __import__('1-minor').minor
determinant = __import__('0-determinant').determinant


def cofactor(matrix):
    """a function def cofactor(matrix):
     that calculates the cofactor matrix of a matrix"""
    if (not isinstance(matrix, list) or
            not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")
    n = len(matrix)

    if matrix == [] or matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")

    minor_matrix = minor(matrix)
    cofactor_matrix = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            p = (-1)**(i+j)
            cofactor_matrix[i][j] = p * minor_matrix[i][j]
    return cofactor_matrix
