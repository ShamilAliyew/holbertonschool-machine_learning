#!/usr/bin/env python3
"""a function def slice(df): that takes a pd.DataFrame and"""


def slice(df):
    """a function def slice(df): that takes a pd.DataFrame and"""
    return df[["High", "Low", "Close", "Volume_BTC"]][::60]
