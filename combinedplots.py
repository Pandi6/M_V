# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:43:38 2019

@author: NWUUser
"""

import numpy as np  
import matplotlib.pyplot as plt    
import pandas as pd
import math
import itertools
from functools import reduce
from numpy.random import seed
import random
from numpy.random import rand
import matplotlib



data = pd.read_excel (r'C:\Users\NWUUser\Documents\PDF\python\datas.xlsx', sheet_name='returns')
Mat = data.iloc[:,]
#returns_daily = Mat.pct_change()
log_return = Mat.pct_change()
log_returns = log_return.drop(0, axis=0)
n, m = np.shape(log_returns)
mu = log_returns.mean()*250
mat1 = log_returns.values
proba = 1/(n-1)

Cov = np.zeros((n,m))
for i in range (0,m):
    all_column = []
    for j in range (0,n):
        kal = log_returns.iloc[j][i]-mu[i]
        all_column.append(kal)
        Cov[j,i] = all_column[j]
Trans = Cov.transpose()
#VarCov = proba*np.dot(Trans, Cov)
Cov = log_returns.cov()
VarCov = Cov*250


benchmark = mat1.mean()
w = [1/m] * m
Rp = []

less_porto = []
for j in range (0,n):
    Products = []
    for i in range (0,m):
        asset = log_returns.iloc[j][i]*w[i]
        Products.append(asset)
    porto = sum(Products)
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
        Porto = sum(products)
        Rp.append(Porto)
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

stock = Mat.columns
num_iterations = 3000
simulation_res = np.zeros((2+ len(stock),num_iterations))
Ssimulation_res = np.zeros((2+ len(stock),num_iterations))
#interval = [0.009, 0.0095, 0.010, 0.0105, 0.011, 0.0115, 0.012]
   
for k in range(num_iterations):
    one_vec = np.ones((m))
    #np.random.seed(444)
    mean = np.random.random(())

    #print(mean)
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
    #stand = ((a*mean**2 - 2*b*mean + c)/delta)**1/2
    
    simulation_res[0,k] = mean
    simulation_res[1,k] = portfolio_std_dev
    #simulation_res[2,k] = simulation_res[0,k] / simulation_res[1,k]
    for t in range(len(optimal_weights)):
        simulation_res[t+2,k] = optimal_weights[t]
    
    if np.linalg.det(optimal) > 1e-10:
        Scov_inv = np.linalg.inv(optimal)
    else:
        Scov_inv = np.linalg.pinv(optimal) 
    aS = np.dot(one_vec.T, np.dot(Scov_inv, one_vec))
    bS = np.dot(one_vec.T, np.dot(Scov_inv, mu))
    cS = np.dot(mu.T, np.dot(Scov_inv, mu))
    deltaS = aS * cS - (bS**2)
    l1S = (cS - bS * mean) / deltaS
    l2S = (aS * mean - bS) / deltaS
    Soptimal_weights = np.dot(Scov_inv, (l1S*one_vec.T+ l2S*mu.T))
    Soptimal_weights /= np.sum(Soptimal_weights)
    #mean = np.dot(optimal_weights.T, mu)
    Sportfolio_return = (np.dot(Soptimal_weights, mu))
  
    Sportfolio_std_dev = (np.sqrt(np.dot(Soptimal_weights.T,np.dot(optimal, Soptimal_weights))))
    
    Ssimulation_res[0,k] = Sportfolio_return
    Ssimulation_res[1,k] = Sportfolio_std_dev
    #Ssimulation_res[2,k] = Ssimulation_res[0,k] / Ssimulation_res[1,k]
    for t in range(len(Soptimal_weights)):
        Ssimulation_res[t+2,k] = Soptimal_weights[t]

sim_frame = pd.DataFrame(Ssimulation_res.T,columns=['returns','st_deviation']+ ['Weight' for Weight in stock])

plt.style.use('seaborn-dark')
#sim_frame.plot.scatter(x = 'st_deviation', y = 'returns', c = 'sharpe_ratio', cmap='viridis',figsize=(8, 5), grid=True, label = "Semivariance")
        
        
Vsim_frame = pd.DataFrame(simulation_res.T,columns=['returns','st_deviation']+ ['Weight' for Weight in stock])

plt.style.use('seaborn-dark')
ax = plt.gca()
sim_frame.plot.scatter(x = 'st_deviation', y = 'returns', figsize=(8, 5), grid=True, label = "Semivariance", ax = ax)
Vsim_frame.plot.scatter(x = 'st_deviation', y = 'returns', color='red',figsize=(8, 5), grid=True, label = "Variance", ax = ax)
#plt.scatter(x=min_variance_port['st_deviation'], y=min_variance_port['returns'], c='red', marker='D', s=130)
#fig = matplotlib.pyplot.figure()
#plt.figure(figsize=((10,8)))
#fig, ax = plt.subplots(1,1, squeeze=False)
#fig, ax = plt.subplots(2)
#ax = ax.flatten()
#ax[0].plt(x = 'st_deviation', y = 'returns', c = 'sharpe_ratio', cmap='viridis',grid=True, label = "Semivariance")
#ax[1].plt(x = 'st_deviation', y = 'returns', c = 'sharpe_ratio', cmap='RdYlBu', grid=True, label = "Variance")

plt.xlabel('Portfolio risk')
plt.ylabel('Expected Returns')

plt.show()
