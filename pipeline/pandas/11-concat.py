#!/usr/bin/env python3
"""a function def concat(df1, df2): that takes two pd.DataFrame objects"""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """a function def concat(df1, df2): that takes two pd.DataFrame objects"""
    df1 = index(df1)
    df2 = index(df2)
    df2 = df2.loc[:pd.todate_time(1417411920, units='s')]
    df = pd.concat([df1, df2], keys=["bitstamp", "coinbase"])
    return df
