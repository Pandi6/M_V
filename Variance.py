# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 16:24:49 2019

@author: NWUUser
"""

import numpy as np  
import matplotlib.pyplot as plt    
import pandas as pd
import math
import itertools
from functools import reduce
from sympy import Symbol


"""Import data prices from the excel sheet"""

data = pd.read_excel (r'C:\Users\NWUUser\Documents\PDF\python\datas.xlsx', sheet_name='returns')
Mat = data.iloc[:,]

log_return = Mat.pct_change() # calculate the daily returns from assets prices

#log_return = np.log(Mat / Mat.shift(1))
log_returns = log_return.drop(0, axis=0)
n, m = np.shape(log_returns) # n observations and m assets
mu = log_returns.mean()*250  # calculate the vector mu of expected assets returns and annualised it
prob = 1/(n-1)

Cov = np.zeros((n,m)) # calculate the covariance matrix of returns
for i in range (0,m):
    all_column = []
    for j in range (0,n):
        kal = log_returns.iloc[j][i]-mu[i]
        all_column.append(kal)
        Cov[j,i] = all_column[j]
Trans = Cov.transpose()
#VarCov = prob*np.dot(Trans, Cov)
Cov = log_returns.cov()
VarCov = Cov*250  # annualizing the covariance matrix

stock = Mat.columns
num_iterations = 3000 # Monte carlo number of iterations
simulation_res = np.zeros((3+ len(stock),num_iterations)) # saving return and volatility for each portfolio constructed bellow

"""Using Monte Carlo to simulate many different portfolios"""
"""Analytical expressions of the optimal weight and the minimum variance, by using Lagrange method"""

for k in range(num_iterations):
    one_vec = np.ones((m))  # vector composed of 1
    mean = np.random.random(()) # varying the portfolio expected return to get different portfolios and construct the efficient frontier
    if np.linalg.det(VarCov) > 1e-10:  # assuring that the covariance matrix is inversible
        cov_inv = np.linalg.inv(VarCov)
    else:
        cov_inv = np.linalg.pinv(VarCov) 
    a = np.dot(one_vec.T, np.dot(cov_inv, one_vec))
    b = np.dot(one_vec.T, np.dot(cov_inv, mu))
    c = np.dot(mu.T, np.dot(cov_inv, mu))
    delta = a * c - (b**2)
    l1 = (c - b * mean) / delta
    l2 = (a * mean - b) / delta
    optimal_weights = np.dot(cov_inv, (l1*one_vec.T+ l2*mu.T)) # optimal weight expression
    optimal_weights /= np.sum(optimal_weights) # assuring the weights sum to 1
    mean = np.dot(optimal_weights.T, mu)
    
    portfolio_return = np.dot(optimal_weights.T, mu)
    portfolio_std_dev = np.sqrt(reduce(np.dot, [optimal_weights, VarCov, optimal_weights.T])) # optimal portfolio standard deviation
    
    simulation_res[0,k] = mean
    simulation_res[1,k] = portfolio_std_dev
    simulation_res[2,k] = simulation_res[0,k] / simulation_res[1,k]
    for t in range(len(optimal_weights)):
        simulation_res[t+3,k] = optimal_weights[t]
sim_frame = pd.DataFrame(simulation_res.T,columns=['returns','st_deviation','sharpe_ratio']+ ['Weight' for Weight in stock]) # Making a dataframe

#print (sim_frame.head (5))
#print (sim_frame.tail (5))

"""Showing the portfolio with the minimum variance"""

min_volatility = sim_frame['st_deviation'].min() 
min_variance_port = sim_frame.loc[sim_frame['st_deviation'] == min_volatility]

"""Plotting the efficient frontier"""

plt.style.use('seaborn-dark')
sim_frame.plot.scatter(x = 'st_deviation', y = 'returns', c = 'sharpe_ratio', cmap='RdYlBu',figsize=(8, 5), grid=True, label = "Variance")
plt.scatter(x=min_variance_port['st_deviation'], y=min_variance_port['returns'], c='red', marker='D', s=130) # show the minimum variance portfolio with the vertex in red

plt.xlabel('Standard Deviation')
plt.ylabel('Expected Returns')
 
print(min_variance_port.T) # print the minimum variance portfolio
