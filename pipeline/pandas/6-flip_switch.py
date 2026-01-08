#!/usr/bin/env python3
"""a function def flip_switch(df):
 that takes a pd.DataFrame and sort and transpose"""


def flip_switch(df):
    """a function def flip_switch(df):
     that takes a pd.DataFrame and sort and transpose"""
    updated_df = df.sort_values(by="Timestamp", ascending=False)
    return updated_df.T
