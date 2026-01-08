#!/usr/bin/env python3
"""a function def fill(df): that takes a pd.DataFrame and:
Removes the Weighted_Price column.
Fills missing values in the Close column with the previous row’s value.
Fills missing values in the High, Low, and Open columns with the
corresponding Close value in the same row.
Sets missing values in Volume_(BTC) and Volume_(Currency) to 0"""


def fill(df):
    """a function def fill(df): that takes a pd.DataFrame"""

    updated_df = df.drop(columns=["Weighted_Price"])
    updated_df["Close"].fillna(method="ffill", inplace=True)
    updated_df[["High","Low","Open"]] = updated_df[["High","Low","Open"]].fillna(updated_df["Close"], inplace=True).ffill()
    updated_df["Volume_(BTC)","Volume_(Currency)"].fillna(0, inplace=True)
    return updated_df
