import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

import cvxpy as cp
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

import sys
sys.path.insert(0, r'C:\Users\user\Documents\GitHub\Portfolio-Construction-and-Analysis\python_files')
import functions_1 as fnc
import FactorModelLibForMOOC as fm

def display_factor_loadings(intercept, coefs, factor_names):
    
    loadings = np.insert(coefs, 0, intercept)
    out = pd.DataFrame(loadings, columns=['Regression Results'])
    out = out.transpose()
    names = ['Intercept'] + factor_names
    out.columns = names
    print(out)

def linear_regression(DependentVar, Factors, return_model=False):
    
    lg = LinearRegression(fit_intercept=True)
    lg.fit(Factors, DependentVar)

    display_factor_loadings(lg.intercept_, lg.coef_, Factors.columns.to_list())

    if return_model:
        return lg