#!/usr/bin/env python3
"""a function def array(df):
 that takes a pd.DataFrame as input and performs the following"""
import pandas as pd


def array(df):
    """a function def array(df):
     that takes a pd.DataFrame as input and performs the following"""
    return df[["High", "Close"]].to_numpy()
