import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize


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

def portfolio_return(weights, returns):
    """
    Computes the return on a portfolio from constituent returns and weights
    weights are a numpy array or N x 1 matrix and returns are a numpy array or N x 1 matrix
    """
    return weights.T @ returns

def portfolio_vol(weights, covmat):
    """
    Computes the vol of a portfolio from a covariance matrix and constituent weights
    weights are a numpy array or N x 1 maxtrix and covmat is an N x N matrix
    """
    return (weights.T @ covmat @ weights)**0.5

from scipy.optimize import minimize

def minimize_vol(target_return, er, cov, weight_constraints = 1):
    """
    Find the optimal portfolio weights that achieve the target return by minimizing portfolio volatility.

    Parameters:
        target_return (float): The desired level of return.
        er (np.ndarray): Array of expected returns for each asset in the portfolio.
        cov (np.ndarray): Covariance matrix for the given assets.
        weight_constraints (float, optional): Default is 1. Weight constraint for the optimization. Anything above indicates leveraging.

    Returns:
        np.ndarray: Optimal weights that achieve the target return.
    """
    n = er.shape[0] #number of assets
    init_guess = np.repeat(1/n, n)
    bounds = ((0, weight_constraints),) * n
    #construct the constraints
    weights_sum_to_contraint = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - weight_constraints
    }
    return_is_target = {
        'type': 'eq',
        'args': (er,),
        'fun': lambda weights, er: target_return - portfolio_return(weights, er)
    }
    weights = minimize(portfolio_vol, init_guess,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_contraint,return_is_target),
                       bounds=bounds)
    return weights.x

def msr(riskfree_rate, er, cov, weight_constraints=1):
    """
    Calculates the Maximum Sharpe Ratio (MSR) portfolio using the given risk-free rate, expected returns, and covariance matrix.

    Parameters:
        riskfree_rate (float): The risk-free rate used in calculating the Sharpe ratio.
        er (np.ndarray): Array of expected returns for each asset in the portfolio.
        cov (np.ndarray): Covariance matrix for the given assets.
        weight_constraints (float, optional): Default is 1. Weight constraint for the optimization. Anything above indicates leveraging.

    Returns:
        pd.DataFrame: A DataFrame containing the MSR portfolio weights, portfolio return, volatility, and Sharpe ratio.
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, weight_constraints),) * n
    # construct the constraints
    weights_sum_to_constraint = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - weight_constraints
    }
    def neg_sharpe(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio
        of the given portfolio
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol
    
    weights = minimize(neg_sharpe, init_guess,
                       args=(riskfree_rate, er, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_constraint,),
                       bounds=bounds)
    df = pd.DataFrame({'Weights':[weights.x],
                       'Portfolio Return': portfolio_return(weights.x, er),
                       'Portfolio Volatility': portfolio_vol(weights.x, cov),
                       'Sharpe Ratio': (portfolio_return(weights.x, er)-riskfree_rate)/portfolio_vol(weights.x, cov)})
    return df

def gmv(cov, er=None, riskfree_rate=0, weight_constraints=1):
    """
    Calculates the Global Minimum Variance (GMV) portfolio based on the given covariance matrix and expected returns.

    Parameters:
        cov (np.ndarray): Covariance matrix of the assets.
        er (np.ndarray, optional): Default is None. Expected returns for each asset in the portfolio.
        riskfree_rate (float, optional): Default is 0. Risk-free rate used in calculating the Sharpe ratio.
        weight_constraints (float, optional): Default is 1. Weight constraint for the optimization. Anything above indicates leveraging.

    Returns:
        pd.DataFrame or np.ndarray: If expected returns (er) are not provided, returns the array of GMV portfolio weights.
        If expected returns (er) are provided, returns a DataFrame containing GMV portfolio weights, portfolio return, volatility, and Sharpe ratio.

    """
    n = cov.shape[0]
    weights = msr(0, np.repeat(1, n), cov, weight_constraints).loc[0, 'Weights']
    if er is None:
        return weights
    else:
        df = pd.DataFrame({'Weights':[weights],
                       'Portfolio Return': portfolio_return(weights, er),
                       'Portfolio Volatility': portfolio_vol(weights, cov),
                       'Sharpe Ratio': (portfolio_return(weights, er)-riskfree_rate)/portfolio_vol(weights, cov)})
        return df

def optimal_weights(n_points, er, cov, weight_constraints=1):
    """
    Returns a list of weights that represent a grid of n_points on the efficient frontier
    """
    target_rs = np.linspace(er.min(), er.max()*weight_constraints, n_points)
    weights = [minimize_vol(target_return, er, cov, weight_constraints) for target_return in target_rs]
    return weights

def plot_ef(n_points, er, cov, dataframe=False, weight_constraints=1, style='.-', legend=False, show_msr=False, riskfree_rate=0,
             show_ew=False, show_gmv=False, figsize=(12,6)):
    """
    Plots the Efficient Frontier or returns a DataFrame if specified, based on expected returns and covariance matrix.

    Parameters:
        n_points (int): Number of points on the efficient frontier curve.
        er (np.ndarray): Expected returns of the assets.
        cov (np.ndarray): Covariance matrix of asset returns.
        dataframe (bool, optional): Default is False. If True, returns a DataFrame with portfolio details.
        weight_constraints (float, optional): Default is 1. The leverage factor for weight optimization.
        style (str, optional): Default is '.-'. Style of the efficient frontier line on the plot.
        legend (bool, optional): Default is False. Show the legend on the plot.
        show_msr (bool, optional): Default is False. Show the Maximum Sharpe Ratio portfolio on the plot.
        riskfree_rate (float, optional): Default is 0. Risk-free rate used in Sharpe ratio calculations.
        show_ew (bool, optional): Default is False. Show the Equal Weighted portfolio on the plot.
        show_gmv (bool, optional): Default is False. Show the Global Minimum Variance portfolio on the plot.
        figsize (tuple, optional): Default is (12, 6). Size of the plot.

    Returns:
        None or pd.DataFrame: If dataframe=True, returns a DataFrame containing portfolio weights, returns, 
        volatility, and Sharpe ratio for each point on the efficient frontier.
        Otherwise, returns None.

    """
    weights = optimal_weights(n_points, er, cov, weight_constraints)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    sharpe_ratio = [(ret - riskfree_rate)/vol for ret, vol in zip(rets, vols)]
    if dataframe:
        df = pd.DataFrame({
            "Weights": weights,
            "Returns": rets, 
            "Volatility": vols,
            "Sharpe Ratio": sharpe_ratio
            })
        return df
    else:
        ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
        })
        ax = ef.plot.line(x="Volatility", y="Returns", style=style, legend=legend, figsize=figsize)
        if show_msr:
            ax.set_xlim(left = 0)
            w_msr = msr(riskfree_rate, er, cov, weight_constraints).loc[0, 'Weights']
            r_msr = portfolio_return(w_msr, er)
            vol_msr = portfolio_vol(w_msr, cov)
            #add CML
            cml_x = [0, vol_msr]
            cml_y = [riskfree_rate, r_msr]
            ax.plot(cml_x, cml_y, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=10)
        if show_ew:
            n = er.shape[0]
            w_ew = np.repeat(1/n, n)
            r_ew = portfolio_return(w_ew, er)
            vol_ew = portfolio_vol(w_ew, cov)
            ax.plot([vol_ew], [r_ew], color='goldenrod', marker='o', markersize=10)
        if show_gmv:
            w_gmv = gmv(cov, weight_constraints=weight_constraints)
            r_gmv = portfolio_return(w_gmv, er)
            vol_gmv = portfolio_vol(w_gmv, cov)
            ax.plot([vol_gmv], [r_gmv], color='midnightblue', marker='o', markersize=10)
        return ax

def backtest_portfolio(returns, portfolio_type, periods_per_year, starting_balance, starting_step=None, rolling_period=0, weights_column=False,
                       riskfree_rate=0, weight_constraints=1, reweight_period=1, target_return=None, manual_weights=None,  *args, **kwargs):
    """
    Backtests different types of portfolios on a given set of returns and user-defined criteria.

    Parameters:
        returns (pd.DataFrame or pd.Series): A DataFrame or Series containing asset returns over time.
        portfolio_type (str): The type of portfolio to backtest ('GMV', 'MSR', 'TR', 'EW', 'manual').
        periods_per_year (int): The number of periods in a year for the given returns dataset.
        starting_balance (float): The initial balance of the investment account.
        starting_step (int or None, optional): The number of starting rows to calculate the initial covariance matrix.
            Required for 'MSR', 'GMV', and 'TR' portfolios, optional for 'EW' portfolio.
        rolling_period (int, optional): The rolling period for calculating covariance matrix. Default is 0 (use all available data).
        weights_column (bool, optional): Whether to include a weight column in the resulting DataFrame. Default is False.
        riskfree_rate (float, optional): The risk-free rate used in calculating the MSR portfolio. Default is 0.
        weight_constraints (float, optional): Weight constraint for portfolio optimization. Default is 1.
        reweight_period (int, optional): The period after which to update the covariance matrix and expected returns.
            Default is 1 (update every period).
        target_return (float or None, optional): Required for 'TR' portfolio. The targeted return for the portfolio. It will  be adjusted according to riskfree_rate
        manual_weights (list or numpy array or None, optional): List of weights. Required if portfolio type is 'manual (Manual Weights)
    Returns:
        pd.DataFrame: A DataFrame containing account value, returns, drawdowns, and optionally weights (if 'weights_column' is True).

    Note:
        The function supports backtesting of four portfolio types: 'GMV' (Global Minimum Variance), 'MSR' (Maximum Sharpe Ratio),
        'TR' (Targeted Return), and 'EW' (Equally Weighted).
    """
    if starting_step is None and (portfolio_type =='MSR' or portfolio_type == 'GMV' or portfolio_type == 'TR'):
        raise ValueError(f"Starting point cannot be None for {portfolio_type}")
    if portfolio_type not in ['EW', 'MSR', 'GMV', 'TR', 'manual']:
        raise ValueError(f"{portfolio_type} is not a valid portfolio type")
    if portfolio_type == 'TR' and target_return is None:
        raise ValueError("target_return cannot be None for Targeted Return Portfolio (TR)")
    if portfolio_type == 'manual' and manual_weights is None:
        raise ValueError(f"manual_weights cannot be None for Manual Weights Portfolio")
    dates = returns.index
    num_steps = len(dates)
    account_value = starting_balance
    if isinstance(returns, pd.Series):
        returns = pd.DataFrame(returns, columns=['R'])
    account_history = pd.DataFrame(index=dates)
    account_history[f'Account Value {portfolio_type}'] = None
    if weights_column:
        account_history['Weights'] = None
    n_col = len(returns.columns)
    weights = np.repeat(1/10000000000, n_col)
    prev_cov = None
    prev_expected_rets = None
    for step in range(num_steps):
        if starting_step is not None and step >= starting_step and step >= rolling_period and step % reweight_period == 0:
            if rolling_period > 0:
                cov = returns.iloc[step - rolling_period + 1:step + 1].cov()
                expected_rets = annualize_returns(returns.iloc[step - rolling_period + 1:step + 1], periods_per_year)
                prev_cov = cov
                prev_expected_rets = expected_rets
            else:
                if prev_cov is not None and prev_expected_rets is not None:
                    cov = returns.iloc[:step].cov()
                    expected_rets = annualize_returns(returns.iloc[:step], periods_per_year)
                    prev_cov = cov
                    prev_expected_rets = expected_rets
                else:
                    # Handle the case when no previous values are available yet
                    cov = returns.iloc[:step].cov()
                    if len(returns.iloc[:step]) < 1:
                        expected_rets = annualize_returns(returns.iloc[:step + 1], periods_per_year)
                    else:
                        expected_rets = annualize_returns(returns.iloc[:step], periods_per_year)
                    prev_cov = cov
                    prev_expected_rets = expected_rets

            if portfolio_type == 'GMV':
                weights = gmv(cov, weight_constraints=weight_constraints)
            elif portfolio_type == 'MSR':
                weights = msr(riskfree_rate, expected_rets, cov, weight_constraints).loc[0, 'Weights']
            elif portfolio_type == 'TR':
                weights = minimize_vol((target_return + riskfree_rate), expected_rets, cov, weight_constraints=weight_constraints)
            elif portfolio_type == 'EW':
                n = len(returns.columns)
                weights = np.repeat(1/n, n)
            elif portfolio_type == 'manual':
                weights = np.array(manual_weights)

        portfolio_returns = portfolio_return(weights, returns.iloc[step])
        account_value *= 1 + portfolio_returns

        account_history[f'Account Value {portfolio_type}'].iloc[step] = account_value
        if weights_column:
            account_history['Weights'].iloc[step] = weights
    dr = drawdown(account_history[[f'Account Value {portfolio_type}']].pct_change())
    backtest_result = {
        'Returns': account_history[f'Account Value {portfolio_type}'].pct_change(),
        f'Account History {portfolio_type}': account_history[f'Account Value {portfolio_type}'],
        'Drawdown': dr[f'Account Value {portfolio_type}', 'Drawdown'],
        'Previous Peak': dr[f'Account Value {portfolio_type}', 'Previous Peak']
    }
    df = pd.DataFrame(backtest_result)
    if weights_column:
        df['Weights'] = account_history['Weights']
    if starting_step != 1:
        if starting_step > rolling_period:
            df = df.iloc[starting_step:]
        else:
            df = df.iloc[rolling_period:]
    return df


def summary_stats(returns, periods_per_year, riskfree_rate=0):
    """
    Returns a summary stats for all the columns in returns
    """
    ann_r = returns.aggregate(annualize_returns, periods_per_year=periods_per_year)
    ann_vol = returns.aggregate(annualize_vol, periods_per_year=periods_per_year)
    ann_sr = returns.aggregate(sharpe_ratio, periods_per_year=periods_per_year, riskfree_rate=riskfree_rate)
    dd = returns.aggregate(lambda returns: drawdown(returns).Drawdown.min())
    skew = returns.aggregate(skewness)
    kurt = returns.aggregate(kurtosis)
    cf_var5 = returns.aggregate(var_gaussian, modified=True)
    hist_cvar = returns.aggregate(cvar_historic)
    df =  pd.DataFrame({
        'Annualized Returns': ann_r,
        'Annualized Vol': ann_vol,
        'Skewness': skew,
        'Kurtosis': kurt,
        'Cornish-Fisher VaR (5%)': cf_var5,
        'Historic Cvar (5%)': hist_cvar,
        'Sharpe Ratio': ann_sr,
        'Max Drawdown': dd
    })
    df['start dates'] = start_dates(returns)
    df['end dates'] = returns.index[-1]
    df['Time Period'] = df['start dates'] - df['end dates']
    df = df.drop(columns=['start dates', 'end dates'])
    return df

def combined_backtesting_result(r, portfolios, periods_per_year, rolling_period=0, riskfree_rate=0, weight_constraints=1,
                                 reweight_period=1, starting_step=None, target_returns=None, weights_column=False, starting_balance=1000, *args, **kwargs):
    """
    Combine and analyze backtesting results for multiple portfolios.

    This function calculates and combines the backtesting results for a list of portfolios, such as 
    Equally Weighted (EW), Maximum Sharpe Ratio (MSR), Global Minimum Variance (GMV), and Target 
    Returns (TR) portfolios.

    Parameters:
       r (pd.DataFrame or pd.Series): Returns dataset.
       portfolios (list): List of portfolio types ('MSR', 'EW', 'GMV', 'TR').
       periods_per_year (int): Number of periods in a year for the given returns dataset.
       rolling_period (int, optional): Rolling period for calculating covariance matrix. Default is 0 (use all available data).
       riskfree_rate (float, optional): Risk-free rate used in calculating the MSR portfolio. Default is 0.
       weight_constraints (float, optional): Weight constraint for portfolio optimization. Default is 1.
       reweight_period (int, optional): Period after which to update the covariance matrix and expected returns.
            Default is 1 (update every period).
       starting_step (int or None, optional): Number of starting rows to calculate the initial covariance matrix.
            Required for 'MSR', 'GMV', and 'TR' portfolios, optional for 'EW' portfolio.
       target_returns (list or float, optional): List of target returns for TR portfolio or a float for a single target return.
       weights_column (bool, optional): Include a weight column in the resulting DataFrame. Default is False.
       starting_balance (float, optional): Initial balance of the investment account. Default is 1000
       *args, **kwargs: Additional arguments.

    Returns:
       pd.DataFrame: A multiindex DataFrame containing portfolio types as main columns and backtesting
       results (returns, account history, drawdowns, etc.) as subcolumns.
    """
    if 'TR' in portfolios:
        portfolio_names = {
        'EW': ['Equally Weighted'],
        'MSR': ['Maximum Sharpe Ratio'],
        'GMV': ['Global Minimum Variance'],
        'TR': [f'Target Returns {target_return*100}%' for target_return in target_returns]
    }
    else:
        portfolio_names = {
        'EW': ['Equally Weighted'],
        'MSR': ['Maximum Sharpe Ratio'],
        'GMV': ['Global Minimum Variance']
        }

    if 'TR' in portfolios and target_returns is None:
        raise ValueError("target_returns cannot be None for Target Returns Portfolio (TR)")
    invalid_portfolios = [p for p in portfolios if p not in portfolio_names.keys()]
    if invalid_portfolios:
        raise ValueError(f"{invalid_portfolios} are not valid portfolio types")
    column_names = []
    for portfolio in portfolios:
        column_names.extend(portfolio_names[portfolio])
    
    portfolio_dfs = []

    for portfolio in portfolios:
        if portfolio != 'TR':
            df = backtest_portfolio(r, portfolio_type=portfolio, periods_per_year=periods_per_year, starting_balance=starting_balance, weights_column=weights_column,
                                    starting_step=starting_step, riskfree_rate=riskfree_rate, rolling_period=rolling_period, reweight_period=reweight_period,
                                    weight_constraints=weight_constraints)
            portfolio_dfs.append(df)
        else:
            for target_return in target_returns:
                df = backtest_portfolio(r, portfolio_type=portfolio, periods_per_year=periods_per_year, starting_balance=starting_balance, weights_column=weights_column,
                                    starting_step=starting_step, riskfree_rate=riskfree_rate, rolling_period=rolling_period, reweight_period=reweight_period,
                                    weight_constraints=weight_constraints, target_return=target_return)
                portfolio_dfs.append(df)

    # Combine all individual portfolio DataFrames into a single DataFrame
    result_df = pd.concat(portfolio_dfs, axis=1)

    # Create a MultiIndex for the columns
    multi_index = pd.MultiIndex.from_product([column_names, df.columns])
    result_df.columns = multi_index
    
    return result_df

def weights_change(weights, columns,  *args, **kwargs):
    """
    Returns a Dataframe which contains the assets as columns and weights as rows on a particular time period, or a multi indexed dataframes if there is
    more than one column in the 'weights'

    Parameters:
        weights(pd.DataFrame, pd.Series): Weights Series or Dataframe
        columns(pandas.index, optional): To name the columns in the resulted dataframe    
    Returns:
        pd.DataFrame: A Dataframe containing the given columns as columns and their corresponding weights on each particular instant
    """
    if isinstance(weights, pd.Series):
        weights = pd.DataFrame(weights)
    dfs = []
    if len(weights.columns) == 1:
        weights_df = pd.DataFrame(columns=columns)
        for date in weights.index:
            weight = weights.loc[date]
            weights_df.loc[date] = weight.values[0]
        result_df = weights_df
    else:
        for column in weights.columns:
            weights_df = pd.DataFrame(columns=columns)
            for date in weights.index:
                weight = weights.loc[date]
                weights_df.loc[date] = weight[column]
            dfs.append(weights_df)
        result_df = pd.concat(dfs, axis=1)
        multi_index = pd.MultiIndex.from_product([weights.columns, columns])
        result_df.columns = multi_index
    
    return result_df

