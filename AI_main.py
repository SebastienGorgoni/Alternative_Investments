"""
-----------------------------------------------------------------------
ALTERNATIVE INVESTMENTS

HEC LAUSANNE - AUTUMN 2021

Title: Most Diversified Portfolio Construction

Authors: Sebastien Gorgoni, Alessandro Loury

File Name: AI_main.py
-----------------------------------------------------------------------

Objective:

    The objective create the most diversified portfolio using 3 different asset classes:
        
        - Stocks: SPX
        - Bonds: Bloomberg (Barclays Capital) US Aggregate Bond Index
        - Alternative assets: S&P Goldman Sachs Commodity Index (S&P GSCI)
        
    To understand whether this portfolio allowed to minimize risks, we will also create 3 other types of allocations for comparison purposes: 
       
        - Maximum Sharpe ratio portfolio 
        - Minimum volatility portfolio 
        - Equal-weights portfolio. 
        
    We will present an in-depth performances and risk analysis of all portfolios. 

Data:
    
    - Stocks: SPDR S&P 500 ETF Trust (SPY)
    - Bonds: iShares Core US Aggregate Bond ETF (AGG)
    - Alternative assets: iShares S&P GSCI Commodity-Indexed Trust ETF

This is the main file of the project called "Most Diversified Portfolio Construction". 
It is divided into 3 part:
    
    1) Import Data
    2) Portfolio Construction
        2.1) Most-Diversified Portfolio
        2.2) Maximum Sharpe Ratio Portfolio
        2.3) Equal Weight Portfolio
        2.4) Minimum Volatility Portfolio
        2.5) Comparison of all Portfolio
    3) Risk Analysis

"""


import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns
from tqdm import tqdm
import os

sns.set_theme(style="darkgrid")

#Set the working directory
os.chdir("/Users/sebastiengorgoni/Documents/HEC Master/Semester 5.1/Alternative Investment/Project")
print("Current working directory: {0}".format(os.getcwd()))

from AI_function import criterion_mdp, criterion_SR, risk_historical, cum_prod, perf, MinVol

#Create files in the working directory
if not os.path.isdir('Plot'):
    os.makedirs('Plot')
    
if not os.path.isdir('Output'):
    os.makedirs('Output')

# =============================================================================
# =============================================================================
# 1) Import Data
# =============================================================================
# =============================================================================

start_fin = dt.datetime(2000, 1, 1).date()
end_fin = dt.datetime(2021, 12, 1).date()

start_ptf = '2007-01-01' #earliest data available is mid-2006

"""Collect the ETFs for the Different Indicies"""

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

"""Get the Risk Free Rate"""

# 1-Month London Interbank Offered Rate (LIBOR), based on U.S. Dollar
libor_US = pd.read_excel("FRED_AI.xlsx", sheet_name='1M Libor US')
libor_US = libor_US.set_index('Date')
libor_US = libor_US[(libor_US.index >= '2009-01-01') & (libor_US.index < '2021-11-01')]
libor_US_mean = ((libor_US.mean()/100).values)

risk_free_rate = libor_US_mean.round(1)

"""Performances of all asset classes"""
plt.figure(figsize=(10,7))
plt.plot(cum_prod(agg_returns[start_ptf:]), 'orange', label = 'AGG')
plt.plot(cum_prod(spy_returns[start_ptf:]), 'purple', label = 'SPY')
plt.plot(cum_prod(gsg_returns[start_ptf:]), 'black', label = 'GSG')
plt.legend(loc='upper left', frameon=True)
plt.title("Cumulative Return", fontsize=15)
plt.savefig('Plot/asset_performance.png', dpi=400)
plt.show()
plt.close()

plt.figure(figsize=(7,5))
corr_asset = sns.heatmap(pd.concat([agg_returns[start_ptf:], spy_returns[start_ptf:], gsg_returns[start_ptf:]], axis=1).corr(), annot=True)
plt.title('Correlation of ETFs')
plt.savefig('Plot/asset_corr.png', dpi=400)
plt.show()
plt.close()

perf_agg = perf(agg_returns[start_ptf:], risk_free_rate)
perf_spy = perf(spy_returns[start_ptf:], risk_free_rate)
perf_gsg = perf(gsg_returns[start_ptf:], risk_free_rate)

perf_asset_merged =  pd.concat([perf_agg, perf_spy, perf_gsg], axis=1)
perf_asset_merged.to_latex('Output/perf_asset_merged.tex', column_format = 'lccc', multicolumn_format='c')

# =============================================================================
# =============================================================================
# 2) Portfolio Construction
# =============================================================================
# =============================================================================

# =============================================================================
# 2.1) Most-Diversified Portfolio
# =============================================================================

## Start the optimization
x0 = np.zeros(len(assets_returns.columns))+0.01 # initial values

## Constraint set
constraint_set = ({'type':'eq', 'fun': lambda x: sum(x) - 1})
bounds_set = [(0 , 1) for i in range(len(assets_returns.columns))]

weights_assets_mdp = assets_returns.copy()*0

start_ptf_index = assets_returns[:start_ptf].shape[0]

for row in tqdm(range(1,len(assets_returns))): 
    exp_returns_assets = assets_returns.iloc[:row-1]

    res_mdp = minimize(criterion_mdp, x0, args=(exp_returns_assets), bounds=bounds_set, method='SLSQP', constraints=constraint_set)
    weights_assets_mdp.iloc[row] = res_mdp.x

ptf_mdp_returns = np.multiply(assets_returns[start_ptf:], weights_assets_mdp[start_ptf:]).sum(1)
ptf_mdp_returns.name = 'MDP'

## Compute Results
perf_ptf_mdp = perf(ptf_mdp_returns, risk_free_rate)

weights_assets_mdp[start_ptf:].plot(figsize=(10, 7))
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Plot/weights_mdp.png', dpi=400, bbox_inches='tight')
plt.show()
plt.close()

plt.plot(cum_prod(ptf_mdp_returns), 'g', label = 'MDP')
plt.plot(cum_prod(agg_returns[start_ptf:]), 'orange', label = 'AGG')
plt.plot(cum_prod(spy_returns[start_ptf:]), 'purple', label = 'SPY')
plt.plot(cum_prod(gsg_returns[start_ptf:]), 'black', label = 'GSG')
plt.legend(loc='upper left', frameon=True)
plt.title("Cumulative Return", fontsize=15)
plt.show()
plt.close()

# =============================================================================
# 2.2) Maximum Sharpe Ratio Portfolio
# =============================================================================

## Start the optimization
x0 = np.zeros(len(assets_returns.columns))+0.01 # initial values

## Constraint set
constraint_set = ({'type':'eq', 'fun': lambda x: sum(x) - 1})

bounds_set = [(0 , 1) for i in range(len(assets_returns.columns))] #Maximum 50% allocation to avoid concentration in periods of crisis

weights_assets_sr = assets_returns.copy()*0

for row in tqdm(range(start_ptf_index, len(assets_returns))): 
    exp_returns_assets = assets_returns.iloc[:row-1]

    res_sr = minimize(criterion_SR, x0, args=(exp_returns_assets), bounds=bounds_set, method='SLSQP', constraints=constraint_set)
    weights_assets_sr.iloc[row] = res_sr.x

ptf_sr_returns = np.multiply(assets_returns[start_ptf:], weights_assets_sr[start_ptf:]).sum(1)
ptf_sr_returns.name = 'SR (Without Views)'

## Compute Results
perf_ptf_sr = perf(ptf_sr_returns, risk_free_rate)

weights_assets_sr[start_ptf:].plot(figsize=(10, 7))
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Plot/weights_sr.png', dpi=400, bbox_inches='tight')
plt.show()
plt.close()

plt.plot(cum_prod(ptf_sr_returns), 'g', label = 'SR (Without Views)')
plt.plot(cum_prod(agg_returns[start_ptf:]), 'orange', label = 'AGG')
plt.plot(cum_prod(spy_returns[start_ptf:]), 'purple', label = 'SPY')
plt.plot(cum_prod(gsg_returns[start_ptf:]), 'black', label = 'GSG')
plt.legend(loc='upper left', frameon=True)
plt.title("Cumulative Return", fontsize=15)
plt.show()
plt.close()

# =============================================================================
# 2.2) Maximum Sharpe Ratio Portfolio (with Views)
# =============================================================================

## Start the optimization
x0 = np.zeros(len(assets_returns.columns))+0.01 # initial values

## Constraint set
constraint_set = ({'type':'eq', 'fun': lambda x: sum(x) - 1})

bounds_set = [(0, 0.6), (0, 0.6), (0, 0.3)]

weights_assets_sr_views = assets_returns.copy()*0

for row in tqdm(range(start_ptf_index, len(assets_returns))): 
    exp_returns_assets = assets_returns.iloc[:row-1]

    res_sr_views = minimize(criterion_SR, x0, args=(exp_returns_assets), bounds=bounds_set, method='SLSQP', constraints=constraint_set)
    weights_assets_sr_views.iloc[row] = res_sr_views.x

ptf_sr_views_returns = np.multiply(assets_returns[start_ptf:], weights_assets_sr_views[start_ptf:]).sum(1)
ptf_sr_views_returns.name = 'SR (With Views)'

## Compute Results
perf_ptf_sr_views = perf(ptf_sr_views_returns, risk_free_rate)

weights_assets_sr_views[start_ptf:].plot(figsize=(10, 7))
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Plot/weights_sr_views.png', dpi=400, bbox_inches='tight')
plt.show()
plt.close()

plt.plot(cum_prod(ptf_sr_views_returns), 'g', label = 'SR (With Views)')
plt.plot(cum_prod(agg_returns[start_ptf:]), 'orange', label = 'AGG')
plt.plot(cum_prod(spy_returns[start_ptf:]), 'purple', label = 'SPY')
plt.plot(cum_prod(gsg_returns[start_ptf:]), 'black', label = 'GSG')
plt.legend(loc='upper left', frameon=True)
plt.title("Cumulative Return", fontsize=15)
plt.show()
plt.close()


# =============================================================================
# 2.3) Equal Weight Portfolio
# =============================================================================

ptf_ew_returns = assets_returns[start_ptf:].mean(axis=1).shift(1)
ptf_ew_returns.name = 'EW'

## Compute Results
perf_ptf_ew = perf(ptf_ew_returns, risk_free_rate)

plt.plot(cum_prod(ptf_ew_returns), 'g', label = 'EW')
plt.plot(cum_prod(agg_returns[start_ptf:]), 'orange', label = 'AGG')
plt.plot(cum_prod(spy_returns[start_ptf:]), 'purple', label = 'SPY')
plt.plot(cum_prod(gsg_returns[start_ptf:]), 'black', label = 'GSG')
plt.legend(loc='upper left', frameon=True)
plt.title("Cumulative Return", fontsize=15)
plt.show()
plt.close()

# =============================================================================
# 2.4) Minimum Volatility Portfolio
# =============================================================================

## Start the optimization
x0 = np.zeros(len(assets_returns.columns))+0.01 # initial values

## Constraint set
constraint_set = ({'type':'eq', 'fun': lambda x: sum(x) - 1})
bounds_set = [(0 , 1) for i in range(len(assets_returns.columns))]

weights_assets_mv = assets_returns.copy()*0

for row in tqdm(range(start_ptf_index, len(assets_returns))): 
    exp_returns_assets = assets_returns.iloc[:row-1]

    res_mv = minimize(MinVol, x0, args=(exp_returns_assets), bounds=bounds_set, method='SLSQP', constraints=constraint_set)
    weights_assets_mv.iloc[row] = res_mv.x

ptf_mv_returns = np.multiply(assets_returns[start_ptf:], weights_assets_mv[start_ptf:]).sum(1)
ptf_mv_returns.name = 'MV'

## Compute Results
perf_ptf_mv = perf(ptf_mv_returns, risk_free_rate)

weights_assets_mv[start_ptf:].plot(figsize=(10, 7))
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Plot/weights_mv.png', dpi=400, bbox_inches='tight')
plt.show()
plt.close()

# =============================================================================
# 2.5) Comparison of All Portfolio
# =============================================================================

perf_ptf_merged = pd.concat([perf_ptf_mdp, perf_ptf_sr, perf_ptf_sr_views, perf_ptf_mv, perf_ptf_ew], axis=1)
perf_ptf_merged.to_latex('Output/perf_ptf_merged.tex', column_format = 'lccccc', multicolumn_format='c')

plt.figure(figsize=(10,7))
plt.plot(cum_prod(ptf_mdp_returns), 'r', label = 'MDP')
plt.plot(cum_prod(ptf_sr_returns), 'b', label = 'SR (Without Views)')
plt.plot(cum_prod(ptf_sr_views_returns), 'c', label = 'SR (With Views)')
plt.plot(cum_prod(ptf_ew_returns), 'g', label = 'EW')
plt.plot(cum_prod(ptf_mv_returns),'m' , label='MV')
plt.legend(loc='upper left', frameon=True)
plt.title("Cumulative Return", fontsize=15)
plt.savefig('Plot/ptf_performances.png', dpi=400)
plt.show()
plt.close()

plt.figure(figsize=(10,7))
plt.plot(cum_prod(agg_returns[start_ptf:]), 'orange', label = 'AGG')
plt.plot(cum_prod(spy_returns[start_ptf:]), 'purple', label = 'SPY')
plt.plot(cum_prod(gsg_returns[start_ptf:]), 'black', label = 'GSG')
plt.plot(cum_prod(ptf_mdp_returns), 'r', label = 'MDP')
plt.plot(cum_prod(ptf_sr_returns), 'b', label = 'SR (Without Views)')
plt.plot(cum_prod(ptf_sr_views_returns), 'c', label = 'SR (With Views)')
plt.plot(cum_prod(ptf_ew_returns), 'g', label = 'EW')
plt.plot(cum_prod(ptf_mv_returns),'m' , label='MV')
plt.legend(loc='upper left', frameon=True)
plt.title("Cumulative Return", fontsize=15)
plt.savefig('Plot/ptf_asset_performance.png', dpi=400)
plt.show()
plt.close()

weights_ptf_merged = pd.DataFrame({'MDP': weights_assets_mdp[start_ptf:].mean(), 
                                   'SR (Without Views)': weights_assets_sr[start_ptf:].mean(),
                                   'SR (With Views)': weights_assets_sr_views[start_ptf:].mean(),
                                   'MV': weights_assets_mv[start_ptf:].mean()})
weights_ptf_merged['EW'] = [1/3, 1/3, 1/3]
weights_ptf_merged.round(3).to_latex('Output/weights_merged.tex', column_format = 'lccccc', multicolumn_format='c')

# =============================================================================
# =============================================================================
# 3) Risk Analysis
# =============================================================================
# =============================================================================

"""30-Day Risk Analysis"""
risk_30d_agg = risk_historical(agg_returns, 0.95, 30)
risk_30d_agg.plot()

risk_30d_spy = risk_historical(spy_returns, 0.95, 30)
risk_30d_spy.plot()

risk_30d_agg = risk_historical(gsg_returns, 0.95, 30)
risk_30d_agg.plot()

risk_30d_mdp = risk_historical(ptf_mdp_returns, 0.95, 30)
risk_30d_mdp.plot(figsize=(10,7))
plt.savefig('Plot/risk_mdp.png', dpi=400, bbox_inches='tight')

risk_30d_sr = risk_historical(ptf_sr_returns, 0.95, 30)
risk_30d_sr.plot(figsize=(10,7))
plt.savefig('Plot/risk_sr.png', dpi=400, bbox_inches='tight')

risk_30d_sr_views = risk_historical(ptf_sr_views_returns, 0.95, 30)
risk_30d_sr_views.plot(figsize=(10,7))
plt.savefig('Plot/risk_sr_views.png', dpi=400, bbox_inches='tight')

risk_30d_ew = risk_historical(ptf_ew_returns, 0.95, 30)
risk_30d_ew.plot(figsize=(10,7))
plt.savefig('Plot/risk_ew.png', dpi=400, bbox_inches='tight')

risk_30d_mv = risk_historical(ptf_mv_returns, 0.95, 30)
risk_30d_mv.plot(figsize=(10,7))
plt.savefig('Plot/risk_mv.png', dpi=400, bbox_inches='tight')
