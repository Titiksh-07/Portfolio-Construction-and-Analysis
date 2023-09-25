import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'python files')
import functions_1 as fnc

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

