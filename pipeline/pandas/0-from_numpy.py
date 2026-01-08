#!/usr/bin/env python3
import pandas as pd


def from_numpy(array):
    n_columns = array.shape[1]
    if n_columns > 26:
        raise ValueError("Number of columns"
                         " cannot exceed 26 for A-Z column names")
    columns = [chr(65+i) for i in range(n_columns)]
    return pd.DataFrame(array, columns=columns)
