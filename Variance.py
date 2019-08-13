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


"""Vector mu for Expected value"""

#data = pd.read_excel (r'C:\Users\NWUUser\Documents\PDF\python\datas.xlsx', sheet_name='returns')
#Mat = data.iloc[:,]
#mat = Mat.values
#n, m = np.shape(mat)

data = pd.read_excel (r'C:\Users\NWUUser\Documents\PDF\python\datas.xlsx', sheet_name='returns')
Mat = data.iloc[:,]
#returns_daily = Mat.pct_change()
log_return = Mat.pct_change()
#print(returns_daily)
#mat1 = returns_daily.values 
#log_return = np.log(Mat / Mat.shift(1))
log_returns = log_return.drop(0, axis=0)
n, m = np.shape(log_returns)
mu = log_returns.mean()*250
prob = 1/(n-1)

Cov = np.zeros((n,m))
for i in range (0,m):
    all_column = []
    for j in range (0,n):
        kal = log_returns.iloc[j][i]-mu[i]
        all_column.append(kal)
        Cov[j,i] = all_column[j]
Trans = Cov.transpose()
#VarCov = prob*np.dot(Trans, Cov)
Cov = log_returns.cov()
VarCov = Cov*250

stock = Mat.columns
num_iterations = 3000
simulation_res = np.zeros((3+ len(stock),num_iterations))
#interval = [0.009, 0.0095, 0.010, 0.0105, 0.011, 0.0115, 0.012]

for k in range(num_iterations):
    one_vec = np.ones((m))
    mean = np.random.random(())
    if np.linalg.det(VarCov) > 1e-10:
        cov_inv = np.linalg.inv(VarCov)
    else:
        cov_inv = np.linalg.pinv(VarCov) 
    a = np.dot(one_vec.T, np.dot(cov_inv, one_vec))
    b = np.dot(one_vec.T, np.dot(cov_inv, mu))
    c = np.dot(mu.T, np.dot(cov_inv, mu))
    delta = a * c - (b**2)
    l1 = (c - b * mean) / delta
    l2 = (a * mean - b) / delta
    optimal_weights = np.dot(cov_inv, (l1*one_vec.T+ l2*mu.T))
    optimal_weights /= np.sum(optimal_weights)
    mean = np.dot(optimal_weights.T, mu)
    
    portfolio_return = np.dot(optimal_weights.T, mu)
    portfolio_std_dev = np.sqrt(reduce(np.dot, [optimal_weights, VarCov, optimal_weights.T]))
    stand = ((a*mean**2 - 2*b*mean + c)/delta)**1/2
    
    simulation_res[0,k] = mean
    simulation_res[1,k] = portfolio_std_dev
    simulation_res[2,k] = simulation_res[0,k] / simulation_res[1,k]
    for t in range(len(optimal_weights)):
        simulation_res[t+3,k] = optimal_weights[t]
sim_frame = pd.DataFrame(simulation_res.T,columns=['returns','st_deviation','sharpe_ratio']+ ['Weight' for Weight in stock])

print (sim_frame.head (5))
print (sim_frame.tail (5))


min_volatility = sim_frame['st_deviation'].min()
min_variance_port = sim_frame.loc[sim_frame['st_deviation'] == min_volatility]

plt.style.use('seaborn-dark')
sim_frame.plot.scatter(x = 'st_deviation', y = 'returns', c = 'sharpe_ratio', cmap='RdYlBu',figsize=(8, 5), grid=True, label = "Variance")
plt.scatter(x=min_variance_port['st_deviation'], y=min_variance_port['returns'], c='red', marker='D', s=130)

plt.xlabel('Standard Deviation')
plt.ylabel('Expected Returns')
 
print(min_variance_port.T)
