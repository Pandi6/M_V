# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:43:51 2019

@author: NWUUser
"""

import numpy as np  
import matplotlib.pyplot as plt    
import pandas as pd
import math
import itertools
from functools import reduce
import random


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

#log_return = np.log(Mat / Mat.shift(1))
log_returns = log_return.drop(0, axis=0)
mat1 = log_returns.values
mu = log_returns.mean()*250
n, m = np.shape(log_returns)

proba = 1/(n-1)
benchmark = mat1.mean()
#print(benchmark)

w = [1/m] * m

Rp = []

less_porto = []
for j in range (0,n):
    products = []
    for i in range (0,m):
        asset = log_returns.iloc[j][i]*w[i]
        products.append(asset)
    porto = sum(products)
    Rp.append(porto)
    if (Rp[j]< benchmark):
        less_porto.append(j)
#print(less_porto) 
#print(benchmark)  
#print(Rp)    
Mo = np.zeros((len(less_porto),m))
for k in range (len(less_porto)):
        k = 0
        while k < len(less_porto): 
            for j in (less_porto):
                row = log_returns.iloc[j]
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
            asset = log_returns.iloc[j][i]*w[i]   
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
                row = log_returns.iloc[j]
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
    M = Mm*250
    
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
#print (np.linalg.eigvals(optimal)>0)
        
  
stock = Mat.columns
#stock = ['A','B','C','D','E']
num_iterations = 3000
simulation_res = np.zeros((3+ len(stock),num_iterations))
#mean = Symbol('mean')
#np.random.seed(101)
for k in range(num_iterations):
    one_vec = np.ones((m))
    mean = np.random.random(())
    #mean = random.randrange(0,1)
    if np.linalg.det(optimal) > 1e-10:
        cov_inv = np.linalg.inv(optimal)
    else:
        cov_inv = np.linalg.pinv(optimal) 
    a = np.dot(one_vec.T, np.dot(cov_inv, one_vec))
    b = np.dot(one_vec.T, np.dot(cov_inv, mu))
    c = np.dot(mu.T, np.dot(cov_inv, mu))
    delta = a * c - (b**2)
    l1 = (c - b * mean) / delta
    l2 = (a * mean - b) / delta
    optimal_weights = np.dot(cov_inv, (l1*one_vec.T+ l2*mu.T))
    optimal_weights /= np.sum(optimal_weights)
    #mean = np.dot(optimal_weights.T, mu)
    portfolio_return = (np.dot(optimal_weights, mu))
    
    #pfolio_returns.append(np.sum(weights * log_returns.mean()) * 250)
    portfolio_std_dev = (np.sqrt(np.dot(optimal_weights.T,np.dot(optimal, optimal_weights))))
    
    
    simulation_res[0,k] = portfolio_return
    simulation_res[1,k] = portfolio_std_dev
    simulation_res[2,k] = simulation_res[0,k] / simulation_res[1,k]
    for t in range(len(optimal_weights)):
        simulation_res[t+3,k] = optimal_weights[t]

sim_frame = pd.DataFrame(simulation_res.T,columns=['returns','st_deviation','sharpe_ratio']+ ['Weight' for Weight in stock])

#+ ['Weight' for Weight in stock]

print (sim_frame.head (5))
print (sim_frame.tail (5))

min_volatility = sim_frame['st_deviation'].min()
min_variance_port = sim_frame.loc[sim_frame['st_deviation'] == min_volatility]


plt.style.use('seaborn-dark')
sim_frame.plot.scatter(x = 'st_deviation', y = 'returns', c = 'sharpe_ratio', cmap='viridis',figsize=(8, 5), grid=True, label = "Semivariance")

#plt.scatter(x=sharpe_portfolio['st_deviation'], y=sharpe_portfolio['returns'], c='red', marker='D', s=200)
plt.scatter(x=min_variance_port['st_deviation'], y=min_variance_port['returns'], c='Blue', marker='D', s=130)

plt.xlabel('Semi deviation')
plt.ylabel('Expected Returns')

#plt.show() 
print(min_variance_port.T)

print (optimal_w)

print ('voila:',BigW)

