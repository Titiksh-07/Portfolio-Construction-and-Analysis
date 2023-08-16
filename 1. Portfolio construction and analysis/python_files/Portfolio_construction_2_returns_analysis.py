import pandas as pd
import numpy as np
from typing import Union
import sys
sys.path.insert(0, 'python_files')
from functions_1 import *

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
    ann_ex_ret = annualize_returns(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
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

def var_historic(r, level=5):
    """
    Calculate the historic Value at Risk (VaR) at a specified confidence level.

    Value at Risk (VaR) measures the maximum potential loss of a portfolio at a given confidence level.

    Parameters:
        r (pd.Series or pd.DataFrame): A Series or DataFrame containing historical asset returns.
        level (int, optional): The confidence level for calculating VaR. Default is 5.

    Returns:
        Union[pd.Series, float]: If input is a DataFrame, returns a Series of VaR values for each asset.
                                If input is a Series, returns a single VaR value.

    Raises:
        TypeError: If the input is not a valid pd.Series or pd.DataFrame.
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def cvar_historic(r, level=5):
    """
    Calculate the Conditional Value at Risk (CVaR) at a specified confidence level.

    Conditional Value at Risk (CVaR) measures the average potential loss of a portfolio beyond the Value at Risk (VaR)
    at a given confidence level.

    Parameters:
        r (pd.Series or pd.DataFrame): A Series or DataFrame containing historical asset returns.
        level (int, optional): The confidence level for calculating CVaR. Default is 5.

    Returns:
        Union[pd.Series, float]: If input is a DataFrame, returns a Series of CVaR values for each asset.
                                If input is a Series, returns a single CVaR value.

    Raises:
        TypeError: If the input is not a valid pd.Series or pd.DataFrame.

    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


from scipy.stats import norm
def var_gaussian(r, level=5, modified=False):
    """
    Calculate the Gaussian Value at Risk (VaR) of a Series or DataFrame.

    If "modified" is True, the modified VaR is returned using the Cornish-Fisher modification.

    Parameters:
        r (pd.Series or pd.DataFrame): A Series or DataFrame containing historical asset returns.
        level (int, optional): The confidence level for calculating VaR. Default is 5.
        modified (bool, optional): If True, the modified VaR using the Cornish-Fisher modification is calculated. Default is False.

    Returns:
        Union[float, pd.Series]: If input is a DataFrame, returns a Series of VaR values for each asset.
                                 If input is a Series, returns a single VaR value.

    """
    # compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 -3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof=0))

from typing import Union

def drawdown(r: Union[pd.Series, pd.DataFrame]):
    """
    Calculate drawdown statistics for asset returns.

    Parameters:
        r (pd.Series or pd.DataFrame): A Series or DataFrame containing historical asset returns.

    Returns:
        Union[pd.DataFrame, pd.Series]: If input is a DataFrame, returns a DataFrame with nested columns for each asset,
                                        containing the wealth index, the previous peaks, and the percentage drawdown.
                                        If input is a Series, returns a DataFrame with wealth index, previous peaks, and drawdown values.

    Example:
        >>> returns_data = pd.read_csv('returns.csv', index_col='Date')
        >>> drawdown_stats = drawdown(returns_data)
        >>> print(drawdown_stats.head())
                              Asset1                       Asset2
                            Wealth Previous Peak Drawdown  Wealth Previous Peak Drawdown
        Date
        2020-01-01  1000.000000   1000.000000  0.000000  1000.000000   1000.000000  0.000000
        2020-01-02  1012.345679   1012.345679  0.000000  1023.456790   1023.456790  0.000000
        2020-01-03  1015.873016   1015.873016  0.000000  1036.812976   1036.812976  0.000000
        2020-01-04  1005.291005   1015.873016 -0.010427  1017.391304   1036.812976 -0.018693
        2020-01-05  1012.658228   1015.873016 -0.003172  1010.416667   1036.812976 -0.025241
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
