#!/usr/bin/env python3
"""a function def slice(df): that takes a pd.DataFrame and"""


def slice(df):
    """a function def slice(df): that takes a pd.DataFrame and"""
    updated_df = df[["High", "Low", "Close", "Volume_BTC"]].iloc[::60]
    updated_df = updated_df.rename(columns={"Volume_BTC": "Volume_(BTC)"})
    return updated_df
