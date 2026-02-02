#!/usr/bin/env python3
"""a function def determinant(matrix):
 that calculates the determinant of a matrix"""


def determinant(matrix):
    """a function def determinant(matrix):
     that calculates the determinant of a matrix"""
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

        # Boş matris (0x0) halı
    if matrix == [[]] or matrix == []:
        return 1

    n = len(matrix)

    # Kvadrat matris yoxlaması
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a square matrix")

    # 1x1 matris üçün determinant
    if n == 1:
        return matrix[0][0]

    # 2x2 matris üçün determinant (hesablamanı sürətləndirmək üçün)
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Rekursiv Laplas açılışı (ilk sətir üzrə)
    det = 0
    for j in range(n):
        # Alt-matrisi (minor) yaratmaq: 0-cı sətir və j-cu sütunu silirik
        sub_matrix = [row[:j] + row[j + 1:] for row in matrix[1:]]
        # İşarəni müəyyən etmək (-1)^(i+j)
        sign = (-1) ** j
        det += sign * matrix[0][j] * determinant(sub_matrix)

    return det
