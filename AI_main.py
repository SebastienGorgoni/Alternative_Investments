#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 14:43:10 2021

@author: sebastiengorgoni
"""
import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
from scipy.optimize import minimize

###Most-Diversified Portfolio###
def criterion_mdp(alloc, Returns):
    """ 
    This function computes the Most-Diversified Portfolio (MDP),
    which attributes the same relative marginal volatility to all the assets.
    
    Parameters
    ----------
    alloc : TYPE
        Weights in the investor's portfolio
    Returns : TYPE
        The returns of the portfolio's assets
    Returns
    -------
    Div_ratio : Object
        Optimal weights of assets in the portfolio.
        
    """
    ptf = np.multiply(Returns.iloc[:,:],alloc)
    ptfReturns = np.sum(ptf,1)
    vol_ptf = np.std(ptfReturns)
    
    numerator = np.multiply(np.std(Returns),alloc)
    numerator = np.sum(numerator)
    
    Div_Ratio = -numerator/vol_ptf
    return Div_Ratio

def cum_prod(returns):
    """
    This function determine the the cumulative returns.

    Parameters
    ----------
    returns : TYPE
        The returns of the asset.

    Returns
    -------
    TYPE
        It returns the cumulative returns.

    """    
    return (returns + 1).cumprod()*100

def hit_ratio(return_dataset):
    """
    This function determine the hit ratio of any time series returns

    Parameters
    ----------
    return_dataset : TYPE
        The returns of the asset.

    Returns
    -------
    TYPE
        It returns the hit ratio.

    """
    return len(return_dataset[return_dataset >= 0]) / len(return_dataset)

def max_drawdown(cum_returns):
    """
    It determines the maximum drawdown over the cumulative returns
    of a time series.

    Parameters
    ----------
    cum_returns : TYPE
        Cumulative Return.

    Returns
    -------
    max_monthly_drawdown : TYPE
        Evolution of the max drawdown (negative output).

    """
    roll_max = cum_returns.cummax()
    monthly_drawdown = cum_returns/roll_max - 1
    max_monthly_drawdown = monthly_drawdown.cummin()
    return max_monthly_drawdown

def perf(returns_ptf, returns_benchmark, name):
    """
    This function compute all the required performances of a time series.
    It also plot the monthly returns, the evolution of the mayx drawdown and 
    the cumulative return of the portfolio vs. benchmark

    Parameters
    ----------
    data : TYPE
        Returns of a given portfolio.
    benchmark : TYPE
        Returns of the benchmark.
    name : TYPE
        Name of the dataframe.
    name_plt : TYPE
        Name given to the plot.

    Returns
    -------
    df : TYPE
        Return a dataframe that contains the annualized returns, volatility,
        Sharpe ratio, max drawdown and hit ratio.

    """
    plt.figure(figsize=(10,7))
    exp = np.mean(returns_ptf,0)*12
    vol = np.std(returns_ptf,0)*np.power(12,0.5)
    sharpe = exp/vol
    max_dd = max_drawdown((returns_ptf+1).cumprod())
    #plt.subplot(121)
    #plt.plot(max_dd, 'g')
    plt.title("Evolution of Max Drawdown", fontsize=15)
    hit = hit_ratio(returns_ptf)
    df = pd.DataFrame({name: [exp, vol, sharpe, max_dd.min(), hit]}, 
                      index = ['Annualized Return', 'Annualized STD', 'Sharpe Ratio', 'Max Drawdown', 'Hit Ratio'])
    #plt.subplot(122)
    plt.plot(cum_prod(returns_ptf), 'b', label=name)
    plt.plot(cum_prod(returns_benchmark), 'r', label='CW Benchmark')
    plt.legend(loc='upper left', frameon=True)
    plt.title("Cumulative Return", fontsize=15)
    #plt.savefig('Plot/'+name+'.png')
    plt.show()
    plt.close()
    return df

start_fin = dt.datetime(2000, 1, 1).date()
end_fin = dt.datetime(2021, 11, 1).date()

#Get Historical Value
agg = yf.download(tickers = 'AGG', start=start_fin, end=end_fin)['Close']
agg.name = 'AGG'
agg_returns = ((agg/agg.shift(1))-1).dropna(how='any')

spy = yf.download(tickers = 'SPY', start=start_fin, end=end_fin)['Close']
spy.name = 'SPY'
spy_returns = ((spy/spy.shift(1))-1).dropna(how='any')

gsg = yf.download(tickers = 'GSG', start=start_fin, end=end_fin)['Close']
gsg.name = 'GSG'
gsg_returns = ((gsg/gsg.shift(1))-1).dropna(how='any')

assets_returns = pd.concat([agg_returns, spy_returns, gsg_returns], axis=1).dropna()

## Start the optimization
x0 = np.zeros(len(assets_returns.columns))+0.01 # initial values

## Constraint set
start_ptf = '2010-01-01'

constraint_set = ({'type':'eq', 'fun': lambda x: sum(x) - 1})
bounds_set = [(0 , 1) for i in range(len(assets_returns.columns))]

weights_assets = assets_returns.copy()*0

for row in range(1,len(assets_returns)): 
    exp_returns_assets = assets_returns.iloc[:row-1]

    res_mdp = minimize(criterion_mdp, x0, args=(exp_returns_assets), bounds=bounds_set, method='SLSQP', constraints=constraint_set)
    weights_assets.iloc[row] = res_mdp.x

mdp_returns = np.multiply(assets_returns[start_ptf:], weights_assets[start_ptf:]).sum(1)

plt.plot(cum_prod(mdp_returns))
plt.plot(cum_prod(agg_returns[start_ptf:]))
plt.plot(cum_prod(spy_returns[start_ptf:]))
plt.plot(cum_prod(gsg_returns[start_ptf:]))


