#!/usr/bin/env python3
import pandas as pd
""" a function def from_file(filename, delimiter):
 that loads data from a file as a pd.DataFrame"""


def from_file(filename, delimiter):
    """ a function def from_file(filename, delimiter):
     that loads data from a file as a pd.DataFrame"""
    df = pd.read_csv(filename, delimiter=delimiter)
    return df
