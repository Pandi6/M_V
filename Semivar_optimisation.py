# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 10:38:44 2019

@author: NWUUser
"""

import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco
import itertools

""" Extracting data (assets prices) from the Excel sheet"""

data = pd.read_excel (r'C:\Users\NWUUser\Documents\PDF\python\datas.xlsx', sheet_name='returns')
Mat = data.iloc[:,]

returns_daily = Mat.pct_change()
returns = returns_daily.drop(0, axis=0)
#log_return = np.log(Mat / Mat.shift(1))
#log_returns = log_return.drop(0, axis=0)
mat1 = returns.values
mu = returns.mean()
n, m = np.shape(returns)

proba = 1/(n-1)
benchmark = mat1.mean()
#print(mu)
#print(benchmark)
w = [1/m] * m

Rp = []

""" Selecting portfolios that underperform the benchmark in the list 'less_porto' """

less_porto = []
for j in range (0,n):
    products = []
    for i in range (0,m):
        asset = returns.iloc[j][i]*w[i]
        products.append(asset)
    porto = sum(products)
    Rp.append(porto)
    if (Rp[j]< benchmark):
        less_porto.append(j)
          
Mo = np.zeros((len(less_porto),m))
for k in range (len(less_porto)):
        k = 0
        while k < len(less_porto): 
            for j in (less_porto):
                row = returns.iloc[j]
                Mo[k,:] = row
                k += 1
#print (Mo)
Semi = np.empty((len(less_porto),m))
for i in range (0,m):
    column = []
    for j in range (0,len(less_porto)):
        less_asset = Mo[:,i]
        diff_target = less_asset[j] - benchmark
        column.append(diff_target)
        Semi[j,i] = column[j]
Semi_trans = Semi.transpose()
Mm = proba*np.dot(Semi_trans, Semi)
M = Mm*250
#print(M)
BigM = [M]
BigW = [w]
x = itertools.count()

for t in x:
    one = np.ones((m))
    if np.linalg.det(BigM[t]) > 1e-10:
        cov_inv = np.linalg.inv(BigM[t])
    else:
        cov_inv = np.linalg.pinv(BigM[t]) 
   
    w = (np.dot(cov_inv, one))/np.dot((np.dot(one.T,cov_inv)), one)
    
    Rp = []
    less_porto = []
    
    for j in range (0,n):
        products = []
        for i in range (0,m):
            asset = returns.iloc[j][i]*w[i]   
            products.append(asset)
        porto = sum(products)
        Rp.append(porto)
        if (Rp[j]< benchmark):
            less_porto.append(j)
        
    Mo = np.zeros((len(less_porto),m))
    for k in range (len(less_porto)):
        k = 0
        while k < len(less_porto): 
            for j in (less_porto):
                row = returns.iloc[j]
                Mo[k,:] = row
                k += 1
    
    Semi = np.zeros((len(less_porto),m))
    for i in range (0,m):
        column = []
        for j in range (0,len(less_porto)):
            less_asset = Mo[:,i]
            diff_target = less_asset[j] - benchmark
            column.append(diff_target)
            Semi[j,i] = column[j]
    Semi_trans = Semi.transpose()
    Mm = proba*np.dot(Semi_trans, Semi)
    M = Mm
    
    BigM.append(M)
    BigW.append(w)
     
    if ((BigM[t] == BigM[t+1]).all()):
        optimal = BigM[t]
        if np.linalg.det(optimal) > 1e-10:
            cov_inv = np.linalg.inv(optimal)
        else:
            cov_inv = np.linalg.pinv(optimal) 
        optimal_w = (np.dot(cov_inv, one))/np.dot((np.dot(one.T,cov_inv)), one)
        break
    
def mean_variance(weights, mean_returns, cov_matrix):
    porto_ret = np.dot(mean_returns, weights.T)*250
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return std_dev, porto_ret


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
   
mean_returns = mu
cov_matrix = optimal*250  
num_portfolios = 1000


results, Weights = random_portfolios(num_portfolios, mean_returns, cov_matrix)


def risk_porto(weights, mean_returns, cov_matrix):
    return mean_variance(weights, mean_returns, cov_matrix)[0]

 
def minimum_var(mean_returns, cov_matrix):
    num_assets = m
    args = (mean_returns, cov_matrix)
    #w = num_assets*[1./num_assets,]
    w = np.array([1. / num_assets for x in range(num_assets)])
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    #bound = (0.0,1.0)
    bounds = tuple((0.,1.) for asset in range(num_assets))

    result = sco.minimize(risk_porto, w , args = args, method = 'SLSQP', bounds = bounds, 
                          constraints = constraints)

    return result  

        
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


def efficient_frontier(mean_returns, cov_matrix, expect_ret):
    efficients = []
    for target in expect_ret:
        efficients.append(optimal_return(mean_returns, cov_matrix, target))
        
    return efficients    


def plot_efficient_frontier(mean_returns, cov_matrix):

    min_volatility = minimum_var(mean_returns, cov_matrix)
    min_r = mean_variance(min_volatility['x'], mean_returns, cov_matrix)[1]
    min_std = mean_variance(min_volatility['x'], mean_returns, cov_matrix)[0]
     
    min_volatility_allocation = pd.DataFrame (min_volatility.x, index = Mat.columns, columns = ['Allocation'])
    
    min_volatility_allocation.Allocation = [round(i*100,2) for i in min_volatility_allocation.Allocation]
    
    min_volatility_allocation = min_volatility_allocation.T
    
    asset_annual_ret = mean_returns * 250
    asset_annual_vol = np.std(returns) * np.sqrt(250)
    
    print ("The global minimum variance:")
    print ("Return:", round(min_r,2))
    print ("Volatility:", round(min_std,2))
    print('\n')
    
    print (min_volatility_allocation )
    print('\n')
    
    print ("Stock Returns and Volatility\n")
    
    for i, asset in enumerate(Mat.columns):
        print (asset,"return:",round(asset_annual_ret[i],2),"and volatility:",round(asset_annual_vol[i],2))
    
    fig, ax = plt.subplots(figsize = (8,5))
    ax.scatter(asset_annual_vol,asset_annual_ret,marker = 'o',s = 100)

    for i, asset in enumerate(Mat.columns):
        ax.annotate(asset, (asset_annual_vol[i],asset_annual_ret[i]), xytext = (-5,5), textcoords = 'offset points')
    #ax.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    ax.scatter(min_std,min_r,marker = '*',color = 'g',s = 100, label = 'Minimum volatility')

    expect_ret = np.linspace(min_r, 0.7, 100)
    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, expect_ret)
    ax.plot([p['fun'] for p in efficient_portfolios], expect_ret, linestyle = '--',
            color = 'black', label='Efficient frontier')
    #ax.set_title('Portfolio Optimization with Individual Stocks')
    ax.set_xlabel('Semideviation (%)')
    ax.set_ylabel('Returns (%)')
    ax.legend(labelspacing = 0.8)

plot_efficient_frontier(mean_returns, cov_matrix)
    
    
    