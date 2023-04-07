###################################################################################################
# Implementation of all strategies.
#
# Every implementation has the same interface
# Input:
#   - single series consisting of columns 
#       unit, rul, s1...sn
#   - window size
# Output: 
#   - part of the series that was determined to be a degradation process
###################################################################################################

import pandas as pd

from scipy.stats import kruskal


def hm_hm(series: pd.DataFrame, window_size: int) -> int:
    test_value_colname = 'tmp_hm_hm_separation'
    test_value_binary = 'tmp_hm_hm_separation_binary'

    diff = 10
    for i, row in series.iterrows():
        if i + window_size - 1 >= series.shape[0] or i < diff:
            series.at[i, test_value_colname] = 0
        else:
            set1 = series.iloc[i-diff: i-diff + window_size]['rul'].to_numpy()
            set2 = series.iloc[i: i + window_size]['forecasted'].to_numpy()

            series.at[i, test_value_colname] = kruskal(set1, set2).pvalue * 100

    diff2 = 25
    for i, row in series.iterrows():
        if i < diff2:
            series.at[i, test_value_binary] = 0
        else:
            yes = max(series.iloc[i - diff2 : i][test_value_colname].to_numpy()) < 5
            series.at[i, test_value_binary] = 1 if yes else 0

    first_jump = 0
    for i, row in series.iterrows():
        if i < 1:
            continue
        else: 
            if series.at[i, test_value_binary] == 1 and series.at[i-1, test_value_binary] == 0:
                first_jump = i - 1
                break
    
    series[test_value_colname] = None
    series[test_value_binary] = None

def hf_hm(series: int) -> int:
    pass

def hf_tm(series: int) -> int:
    pass

def hm_tm(series: int) -> int:
    pass

def hf_tm(series: int) -> int:
    pass
