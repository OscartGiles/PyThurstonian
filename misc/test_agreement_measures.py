# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 11:37:04 2018

@author: pscog
"""

import numpy as np
from simulate_data import simulate_data
import ranking as rk
import matplotlib.pyplot as plt



def A(x):
    """Sample variance of each item"""
    av_x = x.mean(0)
    N = x.shape[0]
    return np.sum(np.square(x - av_x), axis = 0) / (N - 1)

def sra(x):
    """Pooled variance of A(x)"""
    
    return np.sum(A(x)) / x.shape[0]


#Load and preprocess data  
K = 5 #Number of items
J = 30 #Number of participants
L = 1 #Number of trials
C = 1 #Number of conditions

beta = np.array([[0, 0.5, 1]])        

all_sra = []
rand_sra = []
for i in range(1000):
    data = simulate_data(K, J, L, C, beta = beta)
    
    y = data[['Y1', 'Y2', 'Y3']]
    y.to_csv("y_test.csv", index = False)
    
    y = y.values
    
    y_unif = np.array([np.random.permutation(range(3)) + 1 for i in range(J)])
    
    
    all_sra.append(np.sqrt(sra(y)))
    rand_sra.append(np.sqrt(sra(y_unif)))



plt.hist(np.sqrt(rand_sra), color = 'r', alpha = 0.25, normed = True)
plt.hist(np.sqrt(all_sra), color = 'b', alpha = 0.25, normed = True )

#s_l = S(y, 1, 0)
