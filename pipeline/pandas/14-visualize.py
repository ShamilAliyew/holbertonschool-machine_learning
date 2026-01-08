#!/usr/bin/env python3
"""Data Vizualization"""
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df = df.drop(columns=["Weighted_Price"])
df = df.rename(columns={"Timestamp": "Date"})
df["Date"] = pd.to_datetime(df["Date"], unit="s")
df = df.set_index("Date", drop=True)
df["Close"] = pd.fillna(method="ffill", axis=0)
df[["High", "Low", "Open"]] = df[["High", "Low", "Open"]].fillna(df["Close"]).ffill()
df[["Volume_(BTC)", "Volume_(Currency)"]] = df[["Volume_(BTC)", "Volume_(Currency)"]].fillna(0)
print(df)
df_2017 = df.loc["2017":]
daily = df_2017.resample("D").agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume(BTC)': 'sum',
    'Volume(Currency)': 'sum'
})
plt.plot(df_2017["Date"], daily)
plt.legend()
plt.show()