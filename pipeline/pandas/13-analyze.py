#!/usr/bin/env python3
"""a function def analyze(df): that takes a pd.DataFrame and:
Computes descriptive statistics for all columns except
 the Timestamp column"""


def analyze(df):
    """a function def analyze(df): that takes a pd.DataFrame and:
    Computes descriptive statistics for all columns except
     the Timestamp column"""
    return df.drop(columns=["Timestamp"]).describe(include="all")
