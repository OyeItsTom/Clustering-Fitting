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


def Expo(t, scale, growth):
    """
    This code defines a Python function called Expo that 
    takes three input parameters: t scale and growth.
    """
    f = (scale * np.exp(growth * (t-1960)))
    return f


def func(x, k, l, m):
    """
    This code defines a Python function called func that takes four input parameters: x, k, l, and m, the purpose of the 
    function is to return the value of a mathematical function 
    """
    """"Function to use for finding error ranges"""
    k, x, l, m = 0, 0, 0, 0
    return k * np.exp(-(x-l)**2/m)


def err_ranges(x, func, param, sigma):
    """Function to find error ranges for fitted data
    x: x array of the data
    func: defined function above
    param: parameter found by fitted data
    sigma: sigma found by fitted data"""
    import itertools as iter

    low = func(x, *param)
    up = low

    uplow = []
    for p, s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))

    pmix = list(iter.product(*uplow))

    for p in pmix:
        y = func(x, *p)
        low = np.minimum(low, y)
        up = np.maximum(up, y)

    return low, up


def map_corr(df, size=6):
    """Function creates heatmap of correlation matrix for each pair of 
    columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot (in inch)
        
    The function does not have a plt.show() at the end so that the user 
    can savethe figure.
    """

    corr = df.corr()
    plt.figure(figsize=(size, size))
    # fig, ax = plt.subplots()
    plt.matshow(corr, cmap='coolwarm')
    # setting ticks to column names
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.colorbar()
    # no plt.show() at the end


def n_cluster(data_frame):
    '''
    This defines to calculates the sum of squared errors (SSE) for different values of k, the number of clusters, using the K-means clustering algorithm.
    '''
    k_rng = range(1, 10)
    sse = []
    for k in k_rng:
        km = KMeans(n_clusters=k)
        km.fit_predict(data_frame)
        sse.append(km.inertia_)
    return k_rng, sse
