#!/usr/bin/env python3
""" a function def rename(df):
 that takes a pd.DataFrame as input and performs the following"""
import pandas as pd


def rename(df):
    """ a function def rename(df):
     that takes a pd.DataFrame as input and performs the following"""
    df.rename(columns={"Timestamp": "Datetime"}, inplace=True)
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit='s')
    return df
