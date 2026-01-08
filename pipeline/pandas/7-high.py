#!/usr/bin/env python3
"""a function def high(df): that takes a pd.DataFrame and sort"""


def high(df):
    """a function def high(df): that takes a pd.DataFrame and sort"""
    return df.sort_values(by="High", ascending=False)
