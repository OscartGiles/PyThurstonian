# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 15:11:57 2017

@author: pscog
"""


import itertools
import patsy
import pdb
import numpy as np
import pandas as pd
import scipy.stats as sts


from . import ranking as rk


def simulate_data(K, J, L, C, beta = None, scale = None, seed = None):
    """
    args:
        K: The number of items to rank
        J: Number of participants
        L: Number of trials
        C: The number of conditions
        
        beta: Pass the beta matrix. (can be left as None - then beta will be drawn from normal distribution)
        
    """
        
    np.random.seed(seed)
    
    if beta is None:
    
        beta = np.zeros((C, K))
        
        beta[:,1:] = sts.norm.rvs(0, 1, size = beta[:,1:].shape)    
    
        
    conditions = ["C{}".format(i+1) for i in range(C)]
    subs = ["P{}".format(p+1) for p in range(J)]
    trials = ["{}".format(t+1) for t in range(L)]

    data = pd.DataFrame(list(itertools.product(subs, conditions, trials)), columns = ["Subj", "Condition", "Trial"])
    N = data.shape[0]
    
    X = patsy.dmatrix("~0+Condition", data = data)   
    
   
    mu = np.dot(X, beta)
    K = mu.shape[1]
    
    if scale is None:
        scale = sts.lognorm.rvs(1, 0.025, size = J)
    
    scale_all = scale[pd.factorize(data['Subj'])[0]] #Get the scale for every data point

   
    res = sts.norm.rvs(0, 1, size = (N, K-1))
    
    e = (res.T * 1/scale_all).T    
    z = mu.copy()
    z[:,1:] = mu[:, 1:] + e       
  
  
    y = pd.DataFrame(rk.rank(z, axis = -1), columns = ["Y{}".format(i+1) for i in range(z.shape[1])])
    
    return pd.concat((data, y), axis  = 1), scale
    



if __name__ == '__main__':
    
#    d =gen_fake_data(50, seed = 24813285)
    
    K = 4
    J = 10 #Number of participants
    L = 2 #Number of trials
    C = 2 #Number of conditions
    
    d = simulate_data(K, J, L, C, seed = None)
    
