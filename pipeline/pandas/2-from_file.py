#!/usr/bin/env python3
""" a function def from_file(filename, delimiter):
     that loads data from a file as a pd.DataFrame"""
import pandas as pd


def from_file(filename, delimiter):
    """ a function def from_file(filename, delimiter):
     that loads data from a file as a pd.DataFrame"""
    df = pd.read_csv(filename, delimiter=delimiter)
    return df
