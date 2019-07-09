# -*- coding: utf-8 -*-
"""
Created on Tue May 14 09:52:00 2019

@author: NWUUser
"""

import numpy as np
import random
from array import array
import matplotlib.pyplot as plt

""" Short rates under Vasicek model """


def vasicek(r0, a, b, sigma, T, N):    
    dt = T/float(N)    
    rates = [r0]
    for i in range(N):
        dr = a*(b-rates[-1])*dt + sigma*np.random.normal()
        rates.append(rates[-1] + dr)
    return range (N+1), rates

x,y = vasicek(0.04, 0.131, 0.083, 0.037, 2, 200)

print (x,y)   
plt.plot(x,y)
plt.show()
