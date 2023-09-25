import pandas as pd
import numpy as np
import yfinance as yf

def change_timeframe(df, timeframe, aggregation='sum'):
    """
    Resample a DataFrame of returns to a new timeframe.

    This function takes a DataFrame of returns with a datetime index and resamples
    it into the specified timeframe using aggregation.

    Parameters:
        df (DataFrame): The input DataFrame with datetime index.
        timeframe (str): The new timeframe for resampling, e.g., 'D' for daily, 'M' for monthly.
        aggregation (str, optional): Aggregation method to apply during resampling. Default is 'sum'.

    Returns:
        DataFrame: A resampled DataFrame with the specified timeframe and aggregation.

    Example:
        >>> resampled_data = change_timeframe(original_data, 'M', aggregation='mean')
    """
    column_names = df.columns
    aggregation_dict = {column: aggregation for column in column_names}
    resampled_df = df.resample(timeframe).agg(aggregation_dict)
    resampled_df.index = resampled_df.index.to_period(timeframe)
    return resampled_df


def get_returns_data(tickers: list, start=None, end=None, max_period=True, interval='1wk', dividends=True, file_directory=None,
                      replace_tickers=True, index_freq=None):
    """
    Retrieve and process historical returns data for a list of tickers from Yahoo Finance.

    This function fetches historical stock data from Yahoo Finance for the specified tickers
    and time period. It calculates the returns based on closing prices and optionally adjusts for dividends.

    Parameters:
        tickers (list): List of ticker symbols for the desired assets.
        start (str, optional): Start date of the data retrieval period (YYYY-MM-DD).
        end (str, optional): End date of the data retrieval period (YYYY-MM-DD).
        max_period (bool, optional): Use the maximum available data for the given tickers if True.
        interval (str, optional): Interval of the data, e.g., '1d' for daily, '1wk' for weekly.
        dividends (bool, optional): Adjust returns for dividends if True.
        file_directory (str, optional): Save the data as a CSV file in the specified directory.
        replace_tickers (bool, optional): Replace column names with asset names if True.
        index_freq (str, optional): Set the frequency of the resulting dataset index ('D', 'W', 'M', 'Y').

    Returns:
        DataFrame: A DataFrame containing the calculated returns for the specified tickers.

    Example:
        >>> tickers_list = ['SPX', 'AAPL']
        >>> data = get_returns_data(tickers_list, start='2020-01-01', interval='1d', dividends=True)
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
            names = pd.read_csv('Data/cleaned_data/tickers_and_names.csv', index_col='Ticker')
            df.rename(columns={'Close': names.loc[f'{ticker}', 'Asset Name']}, inplace=True)
        else:
            df.rename(columns={'Close': ticker}, inplace=True)
        result_df = pd.concat([result_df, df], axis=1)
    
    if index_freq:
        result_df.index = result_df.index.to_period(index_freq)

    if file_directory:
        result_df.to_csv(file_directory, index=True)
    return result_df


def local_returns_data():
    """
    Returns a dataframe containing the returns of assets
    """
    r = pd.read_csv('Data/cleaned_data/historical_returns_data_1.csv', index_col='Date')
    r = pd.DataFrame(data=r)
    r.index = pd.to_datetime(r.index)
    return r

def start_dates(r):
    """
    Retrieve the start date of historical asset returns data.

    This function takes a DataFrame of historical asset returns and calculates the start date for each asset.
    The returned dictionary contains asset names as keys and their corresponding start dates as values.

    Parameters:
        r (pd.DataFrame): DataFrame containing historical asset returns data.

    Returns:
        dict: A dictionary containing asset names and their respective start dates.
    """
    columns = r.columns.tolist()
    start_dates = {}
    for column in columns:
        index = r[column].dropna().index
        start_date = min(index)
        start_dates[column] = start_date
    sorted_start_dates = dict(sorted(start_dates.items(), key=lambda item: item[1]))
    return sorted_start_dates


def avail_tickers():
    """
    Returns a list of available tickers in the local file 'Data\cleaned_data\tickers_and_names.csv'
    """
    df = pd.read_csv('Data/cleaned_data/tickers_and_names.csv', index_col='Ticker')
    return df.index.to_list()

def annualize_returns(r, periods_per_year):
    """
    Annualizes a set of return
    """

    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1

def annualize_vol(r, periods_per_year):
    """
    Annualizes a set of volatility
    """
    return r.std()*(periods_per_year**0.5)
