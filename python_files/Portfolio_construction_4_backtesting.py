import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'python files')
from functions_1 import *

def backtest_portfolio(returns, portfolio_type, periods_per_year, starting_balance, starting_step=1, rolling_period=0, weights_column=False,
                       riskfree_rate=0, weight_constraints=1, reweight_period=1, target_return=None, manual_weights=None,  *args, **kwargs):
    """
    Backtests different types of portfolios on a given set of returns and user-defined criteria.

    Parameters:
        returns (pd.DataFrame or pd.Series): A DataFrame or Series containing asset returns over time.
        portfolio_type (str): The type of portfolio to backtest ('GMV', 'MSR', 'TR', 'EW', 'manual').
        periods_per_year (int): The number of periods in a year for the given returns dataset.
        starting_balance (float): The initial balance of the investment account.
        starting_step (int , optional): Default is 1. The number of starting rows to calculate the initial covariance matrix.
            Required for 'MSR', 'GMV', and 'TR' portfolios, optional for 'EW' portfolio.
        rolling_period (int, optional): The rolling period for calculating covariance matrix. Default is 0 (use all available data).
        weights_column (bool, optional): Whether to include a weight column as type dictionary in the resulting DataFrame. Default is False.
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
            account_history['Weights'].at[account_history.index[step]] = {col: weight for col, weight in zip(returns.columns, weights)}
    dr = drawdown(account_history[[f'Account Value {portfolio_type}']].pct_change())
    backtest_result = {
        'Returns': account_history[f'Account Value {portfolio_type}'].pct_change(),
        f'Account History': account_history[f'Account Value {portfolio_type}'],
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

def weights_change(weights, *args, **kwargs):
    """
    Returns a Dataframe which contains the assets as columns and weights as rows on a particular time period, or a multi indexed dataframes if there is
    more than one column in the 'weights'

    Parameters:
        weights(pd.DataFrame, pd.Series): Weights Series or Dataframe, each row containig a dictionary of asset name and it's respective weight
    Returns:
        pd.DataFrame: A Dataframe containing the given columns as columns and their corresponding weights on each particular instant
    """
    if isinstance(weights, pd.Series):
        weights = pd.DataFrame(weights)
    dfs = []
    if len(weights.columns) == 1:
        weights_df = pd.DataFrame(columns=weights.iloc[0].keys())
        for date in weights.index:
            weights_df.loc[date] = weights.loc[date]
        result_df = weights_df
    else:
        for column in weights.columns:
            columns = list(weights[column].iloc[0].keys())
            weights_df = pd.DataFrame(columns=columns)
            for date in weights.index:
                weight_dict = weights[column].loc[date]
                weights_df.loc[date] = [weight_dict.get(col, 0.0) for col in columns]
            dfs.append(weights_df)
        
    result_df = pd.concat(dfs, axis=1)
    multi_index = pd.MultiIndex.from_product([weights.columns, columns])
    result_df.columns = multi_index
    
    return result_df
