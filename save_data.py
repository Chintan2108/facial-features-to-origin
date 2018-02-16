# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 15:23:58 2018

@author: asus
"""

import pandas as pd

column_data = ['geo_loc','gender','jaw_ratio', 'nasal_ratio', 'mouth_ratio']
df = pd.DataFrame(columns = column_data)

def addRow(geo_loc, gender, jaw_ratio, nasal_ratio, mouth_ratio): 
    global df
    
    temp_df = pd.DataFrame ([[geo_loc, gender ,jaw_ratio, nasal_ratio, mouth_ratio]], columns = df.columns.values)
    print(temp_df)
    df = df.append(temp_df, ignore_index=True)

def saveDF():
    global df
    
    df.to_csv('data.csv', index=True, header=True, sep=',')

