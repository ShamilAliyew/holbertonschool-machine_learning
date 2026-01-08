#!/usr/bin/env python3
"""a function def from_numpy(array):
 that creates a pd.DataFrame from a np.ndarray"""
import pandas as pd


def from_numpy(array):
    """a function def from_numpy(array):
     that creates a pd.DataFrame from a np.ndarray"""
    n_columns = array.shape[1]
    if n_columns > 26:
        raise ValueError("Number of columns"
                         " cannot exceed 26 for A-Z column names")
    columns = [chr(65+i) for i in range(n_columns)]
    return pd.DataFrame(array, columns=columns)
