import numpy as np
import pandas as pd
from pandas import DataFrame 
from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.pyplot import figure
from statsmodels.tsa.stattools import adfuller

from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler

from IPython.display import display, Markdown, Latex

def moving_avg(df, Target_variable = 'None', n_steps = 6):

    #Declare the array containing the series you want to plot. 
    #For example:
    time_series_array = df[Target_variable].values

    #Compute curves of interest:
    time_series_df = pd.DataFrame(time_series_array)
    smooth_array    = time_series_df.rolling(n_steps).mean()
    
    return (smooth_array) 
    
def return_day(df):
    df['Day'] = df.Day_int
    sec_col = df.pop('Day')
    df.insert(1, 'Day', sec_col)

    df.loc[df['Day'] == 1, 'Day'] = 'Monday'
    df.loc[df['Day'] == 2, 'Day'] = 'Tuesday' 
    df.loc[df['Day'] == 3, 'Day'] = 'Wednesday' 
    df.loc[df['Day'] == 4, 'Day'] = 'Thursday'
    df.loc[df['Day'] == 6, 'Day'] = 'Saturday'
    df.loc[df['Day'] == 5, 'Day'] = 'Friday' 
    df.loc[df['Day'] == 7, 'Day'] = 'Sunday'
    
    return df


def load_files(path_load = 'None', 
               GridofInterest = 1, 
               Number_of_days = 10, 
               cols_to_remove = []):
    
    # Loading file
    # path_load = "C:\Shan office\Data\Telecom Italia\Clean data"
    # GridofInterest = 30;
    # NoofDays = 30;
    NoofFiles = Number_of_days

    for fi1 in range(1,NoofFiles+1):    

        date = "2013-11-"+str(fi1)    
        filenameforload = path_load + "\sms-call-internet-mi-" + date + ".csv";
        dataset = pd.read_csv(filenameforload)    

        # Selecting dataset for only first grid, day 1
        if fi1 == 1:
            df_G1 = dataset[dataset.GridID == GridofInterest]        
            df_G1.insert(0, 'Date', date)
            # df_G1 = pd.concat([df_time, df_G1], axis = 1)
            file_cntr = False
        else:
            df_G1temp = dataset[dataset.GridID == GridofInterest]        
            df_G1temp.insert(0, 'Date', date)
            # df_G1temp = pd.concat([df_time, df_G1temp], axis = 1)
            df_G1 = pd.concat([df_G1, df_G1temp], axis=0)

    print(f'We have collected and combined data for {NoofFiles} days for Grid {GridofInterest}')

    df_G1.reset_index(inplace = True, drop = True)
    # cols_to_remove = ['GridID', 'SMSin', 'SMSout', 'Callin', 'Callout']
    if len(cols_to_remove):
        df_G1.drop(columns= cols_to_remove, inplace = True)

    df = df_G1.copy()
    df = return_day(df)
    df['datetime'] = df['Date'] + '_' +df['Time_stamp'] 
    df['datetime_changed'] = pd.to_datetime(df['datetime'].str.replace('_','T'))
    df.drop(columns=['Date', 'Time_stamp', 'datetime'], inplace = True)
    X = df.pop('datetime_changed')
    df.insert(1, 'Datetime', X)
    
    return (df)


def get_hourly_data(df, datetime_col = 'Datetime', aggregate_time = '60'):
    '''
    aggregate_time: It is 60 by default as I want hourly data aggretation
    '''
    # If datetime_col col is not of type datetime, then do this conversion here.
    # dfx[datetime_col] = pd.to_datetime(dfx[datetime_col])
    dfx = df.copy()
    # First set the datetime data as index
    
    if type(dfx[datetime_col][0]) == str:
        dfx[datetime_col] = pd.to_datetime(dfx[datetime_col])
    
    if type(dfx.index) != pd.core.indexes.datetimes.DatetimeIndex:
        dfx.set_index(datetime_col, inplace = True)
    
    # I am using 60T because I have data captured per 10 minutes. 
    # So there are 6 samples of the data per hour
    aggregate_time = aggregate_time+'T'
    df_hourly = dfx.resample(aggregate_time).sum()
    
    return df_hourly


def stationarity_test(df, Target_variable = 'None'):
    
    '''
    # Stationarity Test

    #### We perform the stationarity test on our data using  Augmented Dicky-Fuller test.

    ### Theory

    The null hypothesis of the test is that the time series can be represented by a unit root, that it is not stationary (has some time-dependent structure). The alternate hypothesis (rejecting the null hypothesis) is that the time series is stationary.

    --><b>Null Hypothesis (H0)</b>: If failed to be rejected, it suggests the time series has a unit root, meaning it is non-stationary. It has some time dependent structure.<br>
    --><b>Alternate Hypothesis (H1)</b>: The null hypothesis is rejected; it suggests the time series does not have a unit root, meaning it is stationary. It does not have time-dependent structure.

    The procedure of the test is that we calculate the p-value from the test. A p-value below a threshold (such as 5% or 1%) suggests we reject the null hypothesis (stationary), otherwise a p-value above the threshold suggests we fail to reject the null hypothesis (non-stationary). 

    -> p-value > 0.05: Fail to reject the null hypothesis (H0), the data has a unit root and is non-stationary. <br>
    -> p-value <= 0.05: Reject the null hypothesis (H0), the data does not have a unit root and is stationary.

    For further details, read [Ref1](https://machinelearningmastery.com/time-series-data-stationary-python/) and [Ref2](https://www.machinelearningplus.com/time-series/augmented-dickey-fuller-test/)
    
    '''
    
    series_activity = df[Target_variable].values
    result = adfuller(series_activity)
    p_val = result[1]
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

    print('\n')
    if p_val <= 0.05:
        print(f'The given series is Stationary, as p-value is {p_val} <= 0.05. ')
        print(f'Furthermore, the ADF Statistic is much smaller than 1\% cricitical value.')
    else:
        print(f'The given series is non-stationary, as p-value is {p_val} > .05')
        
    return 0
            
        
        
    
    