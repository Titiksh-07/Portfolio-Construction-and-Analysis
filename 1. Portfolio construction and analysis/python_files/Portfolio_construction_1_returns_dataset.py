import pandas as pd
import numpy as np


def change_timeframe(df, Timeframe, aggregation='sum'):
    """
    Takes a datetime dataframe of returns and resamples it's every column into the given timeframe

    inputs:
    df - Dataframe
    Timeframe - New timeframe fot the dataset
    aggregation - by default 'sum', but can change according to the need
    """
    column_names = df.columns
    aggregation_dict = {column: aggregation for column in column_names}
    resampled_df = df.resample(Timeframe).agg(aggregation_dict)
    resampled_df.index = resampled_df.index.to_period(Timeframe)
    return resampled_df


import yfinance as yf


def get_returns_data(tickers: list, start=None, end=None, max_period=True, interval='1wk', dividends=True, file_directory=None,
                      replace_tickers=True, index_freq=None):
    """
    Returns a dataframe which contains returns of the mentioned tickers for the mentioned period and interval from yfinance.
    Also has the option to download the data.

    -->Inputs
    tickers: data type(list), takes a list of tickers
    start: default(None), start period of the returns (str)
    end: default(None), end period of the returns (str)
    max_period: default(True), It is the default setting for the function and it gives the maximum available data for the given tickers
    interval: default(1wk), interval of the returns
    dividend: default(True), gives the flexibility to have dividends adjusted returns or not
    file_directory: default(None), downloads the data in csv form at the givend directory
    replace_ticker: default(True), puts column names as asset names instead of tickers, the names are saved locally so this is not applicable
    for any asset
    index_freq: default(None), set the frequency of the resulting dataset to the given frequency ('D', 'W', 'M', 'Y')
    """
    result_df = pd.DataFrame()
    for ticker in tickers:
        obj = yf.Ticker(ticker)
        if max_period and start is None and end is None:
            df = obj.history(period='max', interval=interval)
        else:
            df = obj.history(start=start, end=end, interval=interval)
        if dividends:
            df = df[['Close', 'Dividends']]
            df['Close'] = df['Close'] + df['Dividends'].cumsum()
            df = df.drop(columns='Dividends')
            df = df.pct_change().dropna()
        else:
            df = df[['Close']]
            df = df.pct_change().dropna()
        
        if replace_tickers:
            names = pd.read_csv('Data/cleaned_data/tickers_and_names.csv', index_col = 'Ticker')
            df.rename(columns={'Close':names.loc[f'{ticker}', 'Asset Name']}, inplace=True)
        else:
            df.rename(columns={'Close':ticker}, inplace=True)
        result_df = pd.concat([result_df, df], axis=1)
    
    if index_freq:
        result_df.index = result_df.index.to_period(index_freq)

    if file_directory:
        result_df.to_csv(file_directory, index=True)
    return result_df


def avail_tickers():
    """
    Returns a list of available tickers in the local file 'Data\cleaned_data\tickers_and_names.csv'
    """
    df = pd.read_csv('Data/cleaned_data/tickers_and_names.csv', index_col='Ticker')
    return df.index.to_list()

def annualize_returns(r, periods_per_year):
    """
    Annualizes a set of returns
    """

    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods) - 1

def annualize_vol(r, periods_per_year):
    """
    Annualizes a set of volatility
    """
    return r.std()*(periods_per_year**0.5)
