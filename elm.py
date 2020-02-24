# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 19:19:13 2020

@author: anmol narang
"""
import glob
import os
import pandas as pd

path=r'C:\Users\anmol narang\Desktop\extreme_learning\S5_Yahoo_anomaly_detection_labeled\ydata-labeled-time-series-anomalies-v1_0\A1Benchmark'
all_files = glob.glob(os.path.join(path, "*.csv"))     # advisable to use os.path.join as this makes concatenation OS independent

df_from_each_file = (pd.read_csv(f) for f in all_files)
concatenated_df   = pd.concat(df_from_each_file, ignore_index=True)


