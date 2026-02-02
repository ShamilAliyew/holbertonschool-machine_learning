#!/usr/bin/env python3
"""a function def determinant(matrix):
 that calculates the determinant of a matrix"""


def determinant(matrix):
    """a function def determinant(matrix):
     that calculates the determinant of a matrix"""
    if(not isinstance(matrix, list)) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == [] or matrix == [[]]:
        return 1

    n = len(matrix)

    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a square matrix")

    if n==1:
        return matrix[0][0]

    if n==2:
        return matrix[0][0]*matrix[1][1] -  matrix[0][1]*matrix[1][0]

    det = 0
    for j in range(n):
        sub_matrix = [row[:j] + row[j+1:] for row in matrix[1:]]
        i = (-1)**(0+j)
        det = i * matrix[0][j]*determinant(sub_matrix)

    return det
