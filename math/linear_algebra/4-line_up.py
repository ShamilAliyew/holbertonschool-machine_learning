#!/usr/bin/env python3
"""a function def add_arrays(arr1, arr2):
 that adds two arrays element-wise"""


shape = __import__('2-size_me_please').matrix_shape
def add_arrays(arr1, arr2):
    sum_arr = []
    if shape(arr1) != shape(arr2):
        return None
    else:
        for i in range(len(arr1)):
            sum_arr.append(arr1[i] + arr2[i])
        return sum_arr
