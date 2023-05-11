#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 21:33:15 2023

@author: tomthomas
"""


import numpy as np
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import scipy.optimize as opt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
# %matplotlib inline


def read_data(filename):
    '''This function is used to access the csv dataset.'''
    df = pd.read_excel(filename, skiprows=3)
    return df


def stat_data(df, col, value, yr, a):
    '''
    This function is used to filter data for statistical analysis

    Parameters
    ----------
    df : The dataset received with the data_reading function()
    col : Column name
    value : Value in the selected column
    years : Selected years
    ind : Selected indicator

    Returns
    -------
    df3 : Filtered dataset

    '''
    df3 = df.groupby(col, group_keys=True)
    df3 = df3.get_group(value)
    df3 = df3.reset_index()
    df3.set_index('Indicator Name', inplace=True)
    df3 = df3.loc[:, yr]
    df3 = df3.transpose()
    df3 = df3.loc[:, a]
    df3 = df3.dropna(axis=1)
    return df3
