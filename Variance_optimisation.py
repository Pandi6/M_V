# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 11:45:14 2019

@author: NWUUser
"""


import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco

""" Extracting data (assets prices) from the Excel sheet"""

data = pd.read_excel (r'C:\Users\NWUUser\Documents\PDF\python\datas.xlsx', sheet_name='returns')
Mat = data.iloc[:,]

""" Calculating the assets daily returns"""

returns_daily = Mat.pct_change()
returns = returns_daily.drop(0, axis=0)

n, m = np.shape(returns)

def mean_variance(weights, mean_returns, cov_matrix):
    porto_ret = np.dot(mean_returns, weights.T) *250
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return std_dev, porto_ret

""" Affecting random weights values to each stock"""
""" Defining a list called results to contain every portfolio return and standard deviation
    simulated from the random weights."""

def random_portfolios(num_portfolios, mean_returns, cov_matrix):
    results = np.zeros((2,num_portfolios))
    Weights = []
    for i in range(num_portfolios):
        weights = np.random.random(m)
        weights /= np.sum(weights)
        Weights.append(weights)
        portfolio_return = mean_variance(weights, mean_returns, cov_matrix)[1]
        portfolio_std_dev = mean_variance(weights, mean_returns, cov_matrix)[0]
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        
    return results, Weights
   
mean_returns = returns.mean()
cov_matrix = returns.cov()* 250  # annualised covariance matrix
num_portfolios = 1000


results, Weights = random_portfolios(num_portfolios, mean_returns, cov_matrix)

"""Starting the optimisation"""
""" The function 'risk_porto' returns all the portfolio standard deviation
    simulated above and saved in the first column of the list results. """

def risk_porto(weights, mean_returns, cov_matrix):
    return mean_variance(weights, mean_returns, cov_matrix)[0]

""" Using the optimisation python command 'sco.minimize', solving the portfolio problem by
    minimising the standard deviation given as constraints that the sum of all weights must
    sum to 1 and bound gives limit for each weight value, they must not exceeding the total
    budget neither be negative."""
 
def minimum_var(mean_returns, cov_matrix):
    num_assets = m
    args = (mean_returns, cov_matrix)
    #w = num_assets*[1./num_assets,]
    w = np.array([1. / num_assets for x in range(num_assets)])
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0., 1.)
    bounds = tuple(bound for asset in range(num_assets))

    result = sco.minimize(risk_porto, w , args = args, method = 'SLSQP', bounds = bounds, 
                          constraints = constraints)

    return result  

""" Given an expected portfolio target return defined by 'expect_ret' as second constraint,
    the function 'optimal_return' minimises the portfolio standard deviation subject to
    two constraints now given in 'constraint'. """  
        
def optimal_return(mean_returns, cov_matrix, expect_ret):
    num_assets = m
    args = (mean_returns, cov_matrix)
    #w = num_assets*[1./num_assets,]
    w = np.array([1. / num_assets for x in range(num_assets)])
    
    def portfolio_return(weights):
        return mean_variance(weights, mean_returns, cov_matrix)[1]

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - expect_ret},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0., 1.) for asset in range(num_assets))
    result = sco.minimize(risk_porto, w , args = args, method = 'SLSQP', bounds = bounds,
                          constraints = constraints)
    
    return result

""" Plotting the efficient frontier"""
""" For 100 different expected portfolio returns given in 'expect_ret' below"""

def efficient_frontier(mean_returns, cov_matrix, expect_ret):
    efficients = []
    for target in expect_ret:
        efficients.append(optimal_return(mean_returns, cov_matrix, target))
        
    return efficients    



def plot_efficient_frontier(mean_returns, cov_matrix):

    min_volatility = minimum_var(mean_returns, cov_matrix)
    min_std = mean_variance(min_volatility['x'], mean_returns, cov_matrix)[0]
    min_r = mean_variance(min_volatility['x'], mean_returns, cov_matrix)[1]
    
   
    min_volatility_allocation = pd.DataFrame (min_volatility.x, index = Mat.columns, columns = ['allocation'])
    
    min_volatility_allocation.allocation = [round(i*100,2) for i in min_volatility_allocation.allocation]
    min_volatility_allocation = min_volatility_allocation.T
    
    
    
    print ("The global minimum variance:")
    print ("Return:", round(min_r,2))
    print ("Volatility:", round(min_std,2))
    print('\n')
    
    print (min_volatility_allocation )
    print('\n')
    
    print ("Stock returns and volatility\n")
    
    """ Plotting the efficient and each stock to show their position comparing
    to the minimum variance position and print each stock annualised return and
    volatility. """
    
    asset_annual_ret = mean_returns * 250
    asset_annual_vol = np.std(returns) * np.sqrt(250)
    
    for i, asset in enumerate(Mat.columns):
        print (asset,"return:",round(asset_annual_ret[i],2),"and volatility:",round(asset_annual_vol[i],2))
    
    fig, ax = plt.subplots(figsize = (8,5))
    ax.scatter(asset_annual_vol,asset_annual_ret,marker = 'o',s = 100)

    for i, asset in enumerate(Mat.columns):
        ax.annotate(asset, (asset_annual_vol[i],asset_annual_ret[i]), xytext = (0,5), textcoords = 'offset points')
    #ax.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    ax.scatter(min_std,min_r,marker = '*',color = 'g',s = 100, label = 'Minimum volatility')

    expect_ret = np.linspace(min_r, 0.7, 100)
    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, expect_ret)
    ax.plot([p['fun'] for p in efficient_portfolios], expect_ret, linestyle = '--',
            color = 'black', label='Efficient frontier')
    #ax.set_title('Portfolio Optimization with Individual Stocks')
    ax.set_xlabel('Volatility (%)')
    ax.set_ylabel('Returns (%)')
    ax.legend(labelspacing = 0.8)

plot_efficient_frontier(mean_returns, cov_matrix)

