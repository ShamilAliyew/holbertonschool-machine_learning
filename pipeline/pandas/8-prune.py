#!/usr/bin/env python3
"""a function def prune(df): that takes a pd.DataFrame and:
Removes any entries where Close has NaN values.
"""


def prune(df):
    """a function def prune(df): that takes a pd.DataFrame and:
    Removes any entries where Close has NaN values.
    """

    return df.dropna(subset=["Close"])
