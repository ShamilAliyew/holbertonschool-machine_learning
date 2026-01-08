#!/usr/bin/env python3
"""a function def index(df): that takes a pd.DataFrame and:
Sets the Timestamp column as the index of the dataframe"""
import pandas as pd


def index(df):
    """a function def index(df): that takes a pd.DataFrame and:
    Sets the Timestamp column as the index of the dataframe"""
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit='s')
    df = df.set_index("Timestamp", drop=True)
    return df
