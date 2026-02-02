#!/usr/bin/env python3
"""a function def adjugate(matrix):
 that calculates the adjugate matrix of a matrix"""
cofactor = __import__('2-cofactor').cofactor


def adjugate(matrix):
    """that calculates the adjugate matrix of a matrix"""
    cofactor_matrix = cofactor(matrix)
    adjugate_matrix = []
    for row in zip(*cofactor_matrix):
        adjugate_matrix.append(list(row))
    return adjugate_matrix
