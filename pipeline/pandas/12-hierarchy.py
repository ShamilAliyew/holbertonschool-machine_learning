#!/usr/bin/env python3
"""a function def concat(df1, df2): that takes two pd.DataFrame objects"""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """a function def concat(df1, df2): that takes two pd.DataFrame objects"""
    df1 = index(df1)
    df2 = index(df2)
    updated_df2 = df2.loc[:1417411920]
    df = pd.concat([updated_df2, df1], keys=["bitstamp", "coinbase"])
    return df

def hierarchy(df1, df2):
    """a function def hierarchy(df1, df2): that takes two pd.DataFrame objects"""
    df1 = index(df1)
    df2 = index(df2)
    df = pd.concat([df2, df1], keys=["bitstamp", "coinbase"])
    updated_df = df.swaplevel(0,1)
    updated_df = updated_df.sort_index()

    start = pd.to_datetime(1417411980, unit="s")
    end = pd.to_datetime(1417417980, unit="s")

    return updated_df.loc[start:end]
