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


# Calling the dataset
Mdata = read_data("API_21_DS2_en_excel_v2_5360679.xls")
warnings.filterwarnings("ignore")
start = 1960
end = 2020
year = [str(i) for i in range(start, end+1)]
Indicator = [
    'Merchandise exports to low- and middle-income economies within region (% of total merchandise exports)', 'Trade (% of GDP)']
# creating dataframe data with 2 indicators for clusturing
data = stat_data(Mdata, 'Country Name', 'India', year, Indicator)
Indicator1 = ['Merchandise exports to low- and middle-income economies within region (% of total merchandise exports)',
              'Merchandise imports from low- and middle-income economies within region (% of total merchandise imports)',
              'Trade (% of GDP)', 'Imports of goods and services (% of GDP)',
              'Exports of goods and services (% of GDP)']
# creating dataframe data with 2 indicators for doing correlation
data1 = stat_data(Mdata, 'Country Name', 'India', year, Indicator1)
data = data.rename_axis('Year').reset_index()
data['Year'] = data['Year'].astype('int')
# Renaming colums name to make it short
data1 = data1.rename(columns={
    'Merchandise exports to low- and middle-income economies within region (% of total merchandise exports)': 'Merchandise exports',
    'Merchandise imports from low- and middle-income economies within region (% of total merchandise imports)': 'Merchandise imports',
    'Trade (% of GDP)': 'Trade',
    'Imports of goods and services (% of GDP)': 'Imports(GDP)',
    'Exports of goods and services (% of GDP)': 'Exports(GDP)'})

# This scaler object is then used to transform two columns of a DataFrame, the scaled data columns which can be used for further analysis or modeling.
scaler = MinMaxScaler()
scaler.fit(data[['Trade (% of GDP)']])
data['Scaler_T'] = scaler.transform(
    data['Trade (% of GDP)'].values.reshape(-1, 1))

scaler.fit(
    data[['Merchandise exports to low- and middle-income economies within region (% of total merchandise exports)']])
data['Scaler_M'] = scaler.transform(
    data['Merchandise exports to low- and middle-income economies within region (% of total merchandise exports)'].values.reshape(-1, 1))
data_c = data.loc[:, ['Scaler_T', 'Scaler_M']]
data_c

# This code performs a curve fitting operation
popt, pcov = opt.curve_fit(
    Expo, data['Year'], data['Trade (% of GDP)'], p0=[1000, 0.02])
data["Pop"] = Expo(data['Year'], *popt)
sigma = np.sqrt(np.diag(pcov))
low, up = err_ranges(data["Year"], Expo, popt, sigma)
data2 = data

# This code defines a function that creates a scatter plot.
plt.figure()
sns.scatterplot(data=data2, x="Year", y="Trade (% of GDP)", cmap="Accent")
plt.title('Scatter Plot between 1990-2020 before fitting')
plt.ylabel('Trade (% of GDP)')
# plt.xlabel('Year')
plt.xlim(1990, 2021)
plt.savefig("Scatter_fit.png")
plt.show()


# Plotting the fitted and real data by showing confidence range
plt.figure()
plt.title("Plot After Fitting")
plt.plot(data["Year"], data['Trade (% of GDP)'], label="data")
plt.plot(data["Year"], data["Pop"], label="fit")
plt.fill_between(data["Year"], low, up, alpha=0.7)
plt.legend()
# plt.xlabel("Year")
plt.savefig("Fitted plot.png")
plt.show()

# Used to create a heatmap of the correlation matrix
plt.figure()
corr = data1.corr()
map_corr(data1)
plt.savefig("Heatmap.png")
plt.show()

# This code generates a scatter matrix plot
plt.figure()
pd.plotting.scatter_matrix(data1, figsize=(9, 9))
plt.title("Scatter matrix plot")
plt.tight_layout()
plt.savefig("Scatter matrix.png")
plt.show()


plt.figure()
plt.scatter(data['Trade (% of GDP)'],
            data['Merchandise exports to low- and middle-income economies within region (% of total merchandise exports)'])
#plt.ylabel("Merchandise exports")
# plt.xlabel("Trade")
plt.savefig("Scatter before clusturing.png")
plt.show()

# This code creates a line plot of the sum of squared errors (SSE) for different numbers of clusters
plt.figure()
a, b = n_cluster(data_c)
plt.xlabel = ('k')
plt.ylabel('sum of squared error')
plt.title("Line plot")
plt.plot(a, b)
plt.savefig("Elbow plot.png")
plt.show()

# This code is using k-means clustering algorithm to cluster the data
km = KMeans(n_clusters=2)
pred = km.fit_predict(data_c[['Scaler_T', 'Scaler_M']])
data_c['cludter'] = pred

# To find the center points of the cluster
centers = km.cluster_centers_


# This code segment is creating a scatter plot to visualize the clustering results using KMeans algorithm.
plt.figure()
plt.title("KMean Scatter Plot")
dc1 = data_c[data_c.cludter == 0]
dc2 = data_c[data_c.cludter == 1]
plt.scatter(dc1['Scaler_T'], dc1['Scaler_M'], color='green')
plt.scatter(dc2['Scaler_T'], dc2['Scaler_M'], color='red')
plt.scatter(centers[:, 0], centers[:, 1], s=200, marker='*', color='black')
plt.savefig("Scatter after clusturing.png")
plt.show()


# Predicting future values
low, up = err_ranges(2030, Expo, popt, sigma)
print("Trade (% of GDP) in 2030 is ", low, "and", up)
low, up = err_ranges(2040, Expo, popt, sigma)
print("Trade (% of GDP) in 2040 is ", low, "and", up)

