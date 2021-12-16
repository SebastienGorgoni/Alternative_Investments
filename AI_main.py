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
import seaborn as sns
import os

sns.set_theme(style="darkgrid")

from AI_function import criterion_mdp, criterion_SR, risk_historical, cum_prod, perf

#Set the working directory
os.chdir("/Users/sebastiengorgoni/Documents/HEC Master/Semester 5.1/Alternative Investment/Project")
print("Current working directory: {0}".format(os.getcwd()))

#Create files in the working directory
if not os.path.isdir('Plot'):
    os.makedirs('Plot')
    
if not os.path.isdir('Output'):
    os.makedirs('Output')

# =============================================================================
# 1) Import Data
# =============================================================================

start_fin = dt.datetime(2000, 1, 1).date()
end_fin = dt.datetime(2021, 12, 1).date()

start_ptf = '2010-01-01'

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
# 12-Month London Interbank Offered Rate (LIBOR), based on U.S. Dollar
libor_US = pd.read_excel("FRED_AI.xlsx", sheet_name='12M Libor US')
libor_US = libor_US.set_index('Date')
libor_US = libor_US[(libor_US.index >= '2010-01-01') & (libor_US.index < '2021-11-01')]
libor_US_mean = (libor_US.mean()/100).values

"""Performances of all asset classes"""
plt.figure(figsize=(10,7))
plt.plot(cum_prod(agg_returns[start_ptf:]), 'orange', label = 'AGG')
plt.plot(cum_prod(spy_returns[start_ptf:]), 'purple', label = 'SPY')
plt.plot(cum_prod(gsg_returns[start_ptf:]), 'black', label = 'GSG')
plt.legend(loc='upper left', frameon=True)
plt.title("Cumulative Return", fontsize=15)
plt.savefig('Plot/asset_performance.png')
plt.show()
plt.close()

plt.figure(figsize=(7,5))
corr_asset = sns.heatmap(pd.concat([agg_returns[start_ptf:], spy_returns[start_ptf:], gsg_returns[start_ptf:]], axis=1).corr(), annot=True)
plt.title('Correlation of ETFs')
plt.savefig('Plot/asset_corr.png')
plt.show()
plt.close()

perf_agg = perf(agg_returns[start_ptf:], libor_US_mean)
perf_spy = perf(spy_returns[start_ptf:], libor_US_mean)
perf_gsg = perf(gsg_returns[start_ptf:], libor_US_mean)

perf_asset_merged =  pd.concat([perf_agg, perf_spy, perf_gsg], axis=1)
perf_asset_merged.to_latex('Output/perf_asset_merged.tex')

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

for row in range(1,len(assets_returns)): 
    exp_returns_assets = assets_returns.iloc[:row-1]

    res_mdp = minimize(criterion_mdp, x0, args=(exp_returns_assets), bounds=bounds_set, method='SLSQP', constraints=constraint_set)
    weights_assets_mdp.iloc[row] = res_mdp.x

ptf_mdp_returns = np.multiply(assets_returns[start_ptf:], weights_assets_mdp[start_ptf:]).sum(1)
ptf_mdp_returns.name = 'MDP Portfolio'

## Compute Results
perf_ptf_mdp = perf(ptf_mdp_returns, libor_US_mean)

perf_merged_mdp = pd.concat([perf_agg, perf_spy, perf_gsg, perf_ptf_mdp], axis=1)

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
bounds_set = [(0 , 1) for i in range(len(assets_returns.columns))]

weights_assets_sr = assets_returns.copy()*0

for row in range(start_ptf_index, len(assets_returns)): 
    exp_returns_assets = assets_returns.iloc[:row-1]

    res_mdp = minimize(criterion_SR, x0, args=(exp_returns_assets), bounds=bounds_set, method='SLSQP', constraints=constraint_set)
    weights_assets_sr.iloc[row] = res_mdp.x

ptf_sr_returns = np.multiply(assets_returns[start_ptf:], weights_assets_sr[start_ptf:]).sum(1)
ptf_sr_returns.name = 'SR Portfolio'

## Compute Results
perf_ptf_sr = perf(ptf_sr_returns, libor_US_mean)

perf_merged_sr = pd.concat([perf_agg, perf_spy, perf_gsg, perf_ptf_sr], axis=1)

plt.plot(cum_prod(ptf_sr_returns), 'g', label = 'SR')
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

ptf_ew_returns = assets_returns[start_ptf:].mean(axis=1)
ptf_ew_returns.name = 'EW Portfolio'

## Compute Results
perf_ptf_ew = perf(ptf_ew_returns, libor_US_mean)

perf_merged_ew = pd.concat([perf_agg, perf_spy, perf_gsg, perf_ptf_ew], axis=1)

plt.plot(cum_prod(ptf_ew_returns), 'g', label = 'EW')
plt.plot(cum_prod(agg_returns[start_ptf:]), 'orange', label = 'AGG')
plt.plot(cum_prod(spy_returns[start_ptf:]), 'purple', label = 'SPY')
plt.plot(cum_prod(gsg_returns[start_ptf:]), 'black', label = 'GSG')
plt.legend(loc='upper left', frameon=True)
plt.title("Cumulative Return", fontsize=15)
plt.show()
plt.close()

# =============================================================================
# 2.4) Comparison of All Portfolio
# =============================================================================

perf_ptf_merged = pd.concat([perf_ptf_mdp, perf_ptf_sr, perf_ptf_ew], axis=1)
perf_ptf_merged.to_latex('Output/perf_ptf_merged.tex')

plt.figure(figsize=(10,7))
plt.plot(cum_prod(ptf_mdp_returns), 'r', label = 'MDP')
plt.plot(cum_prod(ptf_sr_returns), 'b', label = 'SR')
plt.plot(cum_prod(ptf_ew_returns), 'g', label = 'EW')
plt.legend(loc='upper left', frameon=True)
plt.title("Cumulative Return", fontsize=15)
plt.savefig('Plot/ptf_performances.png')
plt.show()
plt.close()

plt.figure(figsize=(10,7))
plt.plot(cum_prod(agg_returns[start_ptf:]), 'orange', label = 'AGG')
plt.plot(cum_prod(spy_returns[start_ptf:]), 'purple', label = 'SPY')
plt.plot(cum_prod(gsg_returns[start_ptf:]), 'black', label = 'GSG')
plt.plot(cum_prod(ptf_mdp_returns), 'r', label = 'MDP')
plt.plot(cum_prod(ptf_sr_returns), 'b', label = 'SR')
plt.plot(cum_prod(ptf_ew_returns), 'g', label = 'EW')
plt.legend(loc='upper left', frameon=True)
plt.title("Cumulative Return", fontsize=15)
plt.savefig('Plot/ptf_asset_performance.png')
plt.show()
plt.close()

# =============================================================================
# 3) Risk Analysis
# =============================================================================

"""30-Day Risk Analysis"""
risk_30d_agg = risk_historical(agg_returns, 0.95, 30)
risk_30d_agg.plot()

risk_30d_spy = risk_historical(spy_returns, 0.95, 30)
risk_30d_spy.plot()

risk_30d_agg = risk_historical(gsg_returns, 0.95, 30)
risk_30d_agg.plot()

risk_30d_mdp = risk_historical(ptf_mdp_returns, 0.95, 30)
risk_30d_mdp.plot()

risk_30d_sr = risk_historical(ptf_sr_returns, 0.95, 30)
risk_30d_sr.plot()

risk_30d_ew = risk_historical(ptf_ew_returns, 0.95, 30)
risk_30d_ew.plot()

"""1-Year Risk Analysis"""
risk_1y_agg = risk_historical(agg_returns, 0.95, 252)
risk_1y_agg.plot()

risk_1y_spy = risk_historical(spy_returns, 0.95, 252)
risk_1y_spy.plot()

risk_1y_agg = risk_historical(gsg_returns, 0.95, 252)
risk_1y_agg.plot()

risk_1y_mdp = risk_historical(ptf_mdp_returns, 0.95, 252)
risk_1y_mdp.plot()

risk_1y_sr = risk_historical(ptf_sr_returns, 0.95, 252)
risk_1y_sr.plot()

risk_1y_ew = risk_historical(ptf_ew_returns, 0.95, 252)
risk_1y_ew.plot()