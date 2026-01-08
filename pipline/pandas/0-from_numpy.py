#!/usr/bin/env python3
import pandas as pd
import numpy as np
import string
def from_numpy(array):
    n_columns = array.shape[1]
    columns = list(string.ascii_uppercase[:n_columns])
    return pd.DataFrame(array, columns = columns)
