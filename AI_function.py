"""
-----------------------------------------------------------------------
ALTERNATIVE INVESTMENTS

HEC LAUSANNE - AUTUMN 2021

Title: Most Diversified Portfolio Construction

Authors: Sebastien Gorgoni, Alessandro Loury

File Name: AI_function.py
-----------------------------------------------------------------------

This is an external file for AI_main.py which determine the optimal allocation of all portfolios 
and compute all necessary performances metrics to analyse their performances.

"""
import numpy as np
import pandas as pd
from tqdm import tqdm

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
    ptfReturns = np.sum(ptf, 1)
    vol_ptf = np.std(ptfReturns)
    
    numerator = np.multiply(np.std(Returns),alloc)
    numerator = np.sum(numerator)
    
    Div_Ratio = -numerator/vol_ptf
    return Div_Ratio

###Sharp Ratio Allocation###
def criterion_SR(alloc, Returns, Rf=0):
    """
    This function computes the Maximum Sharpe Ratio Portfolio (SR),
    which attributes the weights that maximize the Sharpe Ratio.
    
    Parameters
    ----------
    alloc : TYPE
        Weights in the investor's portfolio
    Returns : TYPE
        The returns of the portfolio's assets
    Rf : TYPE
        The risk-free rate, assumed to be 0.
        
    Returns
    -------
    SR : Object
        Optimal weights of assets in the portfolio.
    """
    ptf = np.multiply(Returns.iloc[:,:],alloc)
    ptfReturns = np.sum(ptf,1)
    mu_bar = np.mean(ptfReturns)-Rf
    vol_ptf = np.std(ptfReturns)
    SR = -mu_bar/vol_ptf
    return SR

###Minimum Volatility Allocation###
def MinVol(weights, returns):
    """
    This function finds the optimal porfolio with the lowest volatility.

    Parameters
    ----------
    weights : Numpy
        Weights in the investor's portfolio.
    returns : DataFrame
        The returns of the portfolio's assets.

    Returns
    -------
    portfolio_vol : Float
        The volatility of the porfolio.

    """
    performance = np.multiply(returns,weights)
    performance = np.sum(performance,1)
    portfolio_vol = np.std(performance)*np.power(252,.5);
    return portfolio_vol

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

def risk_historical(returns, q, n):
    """
    This function compute the VaR and ES using historical method. 

    Parameters
    ----------
    returns : Dataframe
        The returns of a given strategy, asset, etc.
    q : Integer
        The quantile selected to compute the VaR and ES.
    n : Integer
        The number of months to compute the VaR and ES.

    Returns
    -------
    df : Dataframe
        It returns the evolution of VaR and ES.

    """
    VaR_list = []
    ES_list = []
    for i in tqdm(range(returns.shape[0] - n - 1)):
        temp = - returns[i:n+i].dropna()
        temp_sort = temp.sort_values(ascending=False) #Sort in descending order
        #Var#
        VaR_temp = temp_sort.quantile(q)
        #ES#
        ES_temp = temp[temp > VaR_temp].mean()
        VaR_list.append(VaR_temp)
        ES_list.append(ES_temp)
    
    df = pd.DataFrame({'VaR': VaR_list, 'ES': ES_list}, index=returns[n+1:].index)
        
    return df

def perf(returns_ptf, rf):
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

    exp = np.mean(returns_ptf,0)*252
    vol = np.std(returns_ptf,0)*np.power(252,0.5)
    sharpe = (exp - rf)/vol
    max_dd = max_drawdown((returns_ptf+1).cumprod())
    hit = hit_ratio(returns_ptf)
    skew = returns_ptf.skew()
    excess_kurt = returns_ptf.kurt()
    sterling = (exp - rf)/max_dd.min()
    burke = (exp - rf)/(max_dd**2).sum()
    risk_30d = risk_historical(returns_ptf, 0.95, 30)
    VaR_30d = risk_30d['VaR'].mean()
    ES_30d = risk_30d['ES'].mean()
    # risk_1y = risk_historical(returns_ptf, 0.95, 252)
    # VaR_1y = risk_1y['VaR'].mean()
    # ES_1y = risk_1y['ES'].mean()
    df = pd.DataFrame({'Annualized Return (%)': exp*100, 'Annualized STD (%)': vol*100, 'Sharpe Ratio': sharpe,
                       'Sterling Ratio': sterling, 'Burke Ratio': burke,
                       'Max Drawdown (%)': max_dd.min()*100, 'Hit Ratio (%)': hit*100, 
                       'Skewness': skew, 'Kurtosis': excess_kurt,
                       '30d 95% VaR (%)': VaR_30d*100, '30d 95% ES (%)': ES_30d*100}).T
   
    df.rename(columns={0: returns_ptf.name}, inplace=True)

    return df.round(3)


