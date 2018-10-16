# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 12:27:59 2018

@author: pscog
"""

import numpy as np
import pystan
import pickle
import ranking as rk
import scipy.stats as sts


def ordered_transform(x):
    
    out = np.empty(x.shape)
    
    
    for i in range(x.size):
        
        if i == 0:
            out[i] = x[i]
        
        else:
            out[i] = np.log(x[i] - x[i-1])
            
    return out
        

def inverse_ordered_transform(x):
    
    
    out = np.empty(x.shape)
    
    
    for i in range(x.size):
        
        if i == 0:
            out[i] = x[i]
        
        else:
            out[i] = out[i-1] + np.exp(x[i])
            
    return out            


#a = np.array([0, 0.6, 7, 24])
#
#stan_data = dict(N = 100, K = 5)
##sm = pystan.StanModel(file="simple_transform.stan")
##
##with open("model.pkl", 'wb') as f:
##    pickle.dump(sm, f)
#sm = pickle.load(open('model.pkl', 'rb'))
#
#fit = sm.sampling(data=stan_data, iter=1000, chains=1, control = dict(adapt_delta = 0.999))
#
#la = fit.extract()
#print(np.argsort(la['z_plus']))


N = 10
mu = np.array([0, -2, 1])
K = mu.size
res = sts.norm.rvs(0, 1, size = (N, K-1))   
    
z = np.zeros((N, K))
z[:, 1:] = mu[1:] + res    

y_obs = rk.rank(z, axis = -1)




 
