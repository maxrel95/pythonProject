#!/usr/bin/env python3
########################################################################################################################
# EMF-QARM - Python Workshop - Main
# Authors: Maxime Borel, Coralie Jaunin
# Creation Date: February 27, 2023
# Revised on: February 27, 2023
########################################################################################################################
# load package
import pickle
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd, DateOffset
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter
########################################################################################################################
# load modules
from Code.project_parameters import path, start_date, end_date, firm_sample_ls
#from Code.project_functions import winsorize
########################################################################################################################
# open data
path_to_file = path.get('Inputs') + '/SP500.csv'
sp500_df = pd.read_csv(path_to_file, delimiter=',')
########################################################################################################################
# sample selection
# format date variable
sp500_df['clean_date'] = pd.to_datetime(sp500_df['date'], format='%Y-%m-%d')
# time period
mask = (sp500_df.clean_date >= pd.to_datetime(start_date)) & (sp500_df.clean_date <= pd.to_datetime(end_date))
sp500_subset_df = sp500_df[mask].copy()
# monthly
sp500_subset_df.sort_values(['Name', 'clean_date'], inplace=True)
sp500_subset_df['clean_yearmonth'] = sp500_subset_df.clean_date.dt.to_period('M')
sp500_subset_df = sp500_subset_df.groupby(['clean_yearmonth', 'Name']).last().reset_index()
# firm sample
mask = sp500_subset_df['Name'].astype('category').isin(firm_sample_ls)
sp500_subset_df = sp500_subset_df[mask]
########################################################################################################################
# slice
sp500_subset_close_df = sp500_subset_df.pivot(index=['clean_date'], columns=['Name'],values=['close'])
sp500_subset_close_df.columns = sp500_subset_close_df.columns.get_level_values(1)
sp500_subset_close_df.iloc[:, 0]
sp500_subset_close_df.iloc[4:12, 0]
sp500_subset_close_df.loc[:, 'AAL']
sp500_subset_close_df.loc[pd.to_datetime('2016-06-30'), 'AAL']
sp500_subset_close_df.loc[sp500_subset_close_df.index > pd.to_datetime('2016-06-30'), 'AAL']
sp500_subset_close_df.loc[sp500_subset_close_df.index > pd.to_datetime('2016-06-30'), ['AAL', 'T']]
########################################################################################################################
# returns
sp500_subset_df_copy = sp500_subset_df.copy()
sp500_subset_df['ret'] = sp500_subset_df.groupby('Name', group_keys=False)['close'].apply(lambda s:
                                                                                          np.log(s) - np.log(s.shift(1)))
# what can go wrong?
mask = sp500_subset_df.clean_date != pd.to_datetime('2016-06-30')
sp500_subset_df = sp500_subset_df[mask]
sp500_subset_df['ret_missing'] = sp500_subset_df.groupby('Name', group_keys=False)['close'].apply(
    lambda s: np.log(s) - np.log(s.shift(1)))
# one of many solution
sp500_subset_df = sp500_subset_df_copy.copy()
sp500_subset_df['monthend_date'] = sp500_subset_df.clean_date + MonthEnd(0)
lagged_dates = sp500_subset_df.monthend_date + DateOffset(months=-1) + MonthEnd(0)
lagged_index = pd.MultiIndex.from_arrays([sp500_subset_df['Name'], lagged_dates])
sp500_subset_df = sp500_subset_df.set_index(['Name', 'monthend_date'])
sp500_subset_df['ret'] = np.log(sp500_subset_df.close) - np.log(sp500_subset_df.close.reindex(lagged_index)).values
# what if some observations are missing  --> it still works
sp500_subset_df_copy = sp500_subset_df.copy()
mask = sp500_subset_df.clean_date != pd.to_datetime('2016-06-30')
sp500_subset_df = sp500_subset_df[mask].reset_index()
lagged_dates = sp500_subset_df.monthend_date + DateOffset(months=-1) + MonthEnd(0)
lagged_index = pd.MultiIndex.from_arrays([sp500_subset_df['Name'], lagged_dates])
sp500_subset_df = sp500_subset_df.set_index(['Name', 'monthend_date'])
sp500_subset_df['ret_missing'] = np.log(sp500_subset_df.close) - np.log(sp500_subset_df.close.reindex(lagged_index)).values
print(sp500_subset_df.loc[('AAL', pd.to_datetime('2016-07-31')), ['ret', 'ret_missing']])
# one of many solution using the pivoted dataframe
lagged_index = (sp500_subset_close_df.index + DateOffset(months=-1)).to_period('M')
sp500_subset_close_df.index = sp500_subset_close_df.index.to_period('M')
sp500_subset_ret_df = np.log(sp500_subset_close_df).values - np.log(sp500_subset_close_df.reindex(lagged_index)).values
sp500_subset_ret_df = pd.DataFrame(sp500_subset_ret_df, index=sp500_subset_close_df.index, columns=sp500_subset_close_df.columns)
