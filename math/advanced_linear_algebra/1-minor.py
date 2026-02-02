#!/usr/bin/env python3
"""a function def minor(matrix):
 that calculates the minor matrix of a matrix"""
determinant = __import__('0-determinant').determinant


def minor(matrix):
    """that calculates the minor matrix of a matrix"""
    if (not isinstance(matrix, list) or
            not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")
    n = len (matrix)

    if matrix == [] or matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")

    minor_matrix = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            sub_matrix = [row[:j] + row[j+1:]
                          for row in (matrix[:i] + matrix[i+1:])]
            minor_matrix[i][j] = determinant(sub_matrix)
    return minor_matrix
