import pandas as pd
import numpy as np
from typing import Union
import sys
sys.path.insert(0, 'python_files')
import functions_1 as fnc

def semi_deviation(r):
    """
    Returns the semideviation of the given Series or Datarame of returns
    """
    is_negative = r < 0
    return r[is_negative].std(ddof=0)

def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns
    """
    # convert the annual riskfree rate to per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = fnc.annualize_returns(excess_ret, periods_per_year)
    ann_vol = fnc.annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol

def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3


def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4


def drawdown(r: Union[pd.Series, pd.DataFrame]):
    """
    Takes a DataFrame or series of asset returns.
    Returns a DataFrame with nested columns for each asset, containing the wealth index, the previous peaks, and the percentage drawdown.
    """
    wealth_indices = 1000 * (1 + r).cumprod()
    previous_peaks = wealth_indices.cummax()
    drawdowns = (wealth_indices - previous_peaks) / previous_peaks

    if isinstance(r, pd.DataFrame):
        # Create a MultiIndex for nested columns
        columns = pd.MultiIndex.from_product([r.columns, ["Wealth", "Previous Peak", "Drawdown"]])

        # Create the result DataFrame with the nested columns
        result_df = pd.DataFrame(columns=columns, index=r.index)

        # Fill in the data for each asset
        for column in r.columns:
            result_df[(column, "Wealth")] = wealth_indices[column]
            result_df[(column, "Previous Peak")] = previous_peaks[column]
            result_df[(column, "Drawdown")] = drawdowns[column]
    else:
        wealth_index = 1000*(1+r).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks)/previous_peaks
        return pd.DataFrame({"Wealth": wealth_index, 
                             "Previous Peak": previous_peaks, 
                             "Drawdown": drawdowns})
    
    return result_df