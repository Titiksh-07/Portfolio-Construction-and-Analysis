import pandas as pd
import numpy as np

def get_close_price_df(OHLC_file_path, Column_name = 'Close'):
    """
    Converts an OHLC csv file into a dataframe and then returns the close column along with date column as index
    Best for higher time frame datasets like daily and weekly because it has no time section in Datetime column
    """
    df = pd.read_csv(OHLC_file_path)
    df = df[["Date", "Close"]]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    df = df.rename(columns={'Close':Column_name})

    return df

def clean_dividends_df(file_path, new_column_name):
    """
    Takes a csv file containing dividends release date and dividends amount
    """
    df = pd.read_csv(file_path)
    df = df.set_index('Date')
    df.index = pd.to_datetime(df.index)
    df = df.rename(columns={df.columns[0]:new_column_name})
    return df

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
    return resampled_df

def get_four_major_asset_classes():
    """
    Gives the dividends adjusted returns of four major asset classes -> Gold, Real Estate(VNQ.mx), Bonds(BND.mx) and Equity(VTI.mx)
    """

    GLD = get_close_price_df('Data\OHLC_data\SPDR_Gold_Shares _(GLD).csv', 'Gold')
    VNQ = get_close_price_df('Data\OHLC_data\Vanguard_Real_Estate_Index-Fund_(VNQ).csv', 'Real Estate')
    BND = get_close_price_df('Data\OHLC_data\Vanguard_Total_Bond_Market_Index_Fund_(BND).csv', 'Bonds')
    VTI = get_close_price_df('Data\OHLC_data\Vanguard_Total_Market_Index_Fund_(VTI).csv', 'Equity')

    BND_div = clean_dividends_df('Data\Dividends_data\BND.csv', 'Bonds Div')
    VNQ_div = clean_dividends_df('Data\Dividends_data\VNQ.csv', 'Real Estate Div')
    VTI_div = clean_dividends_df('Data\Dividends_data\VTI.csv', 'Equity Div')

    dfs = [GLD, VNQ, BND, VTI]
    df = pd.concat(dfs, axis=1)
    df = pd.merge_asof(df.sort_index(), change_timeframe(BND_div, 'W').sort_index(),
                          left_index=True, right_on='Date', direction='nearest', tolerance=pd.Timedelta(days=7))
    df = pd.merge_asof(df.sort_index(), change_timeframe(VNQ_div, 'W').sort_index(),
                          left_index=True, right_on='Date', direction='nearest', tolerance=pd.Timedelta(days=7))
    df = pd.merge_asof(df.sort_index(), change_timeframe(VTI_div, 'W').sort_index(),
                          left_index=True, right_on='Date', direction='nearest', tolerance=pd.Timedelta(days=7))
    df.fillna(0, inplace=True)
    df['Real Estate'] = df['Real Estate'] + df['Real Estate Div'].cumsum()
    df['Bonds'] = df['Bonds'] + df['Bonds Div'].cumsum()
    df['Equity'] = df['Equity'] + df['Equity Div'].cumsum()
    df = df.drop(columns=['Real Estate Div', 'Bonds Div', 'Equity Div'])
    df = df.pct_change().dropna()

    return df


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