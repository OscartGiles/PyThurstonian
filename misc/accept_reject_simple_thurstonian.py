 # -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 15:28:36 2018

@author: pscog
"""

import sys
rootpath = 'M:\Transport_Studies\Work_Projects\PSI\Publication_Projects\PyThurstonian'
sys.path.append(rootpath)

from PyThurstonian import thurstonian, simulate_data, run_sample, hdi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from multiprocessing import Process #If we want to use multiprocessing

import pdb





def model(mu):
   
    res = sts.norm.rvs(0, 1, size = (N, K))   
    
    z = mu + res    
    
    y_obs = rk.rank(z, axis = -1)
    
    return y_obs



class normal_prior:
    
    def __init__(self, mu, sigma):
        
        self.norm = sts.norm(mu, sigma)
        
    def rvs(self):
        
        z_k = self.norm.rvs()
        
        return np.array([0, z_k])
    
    def pdf(self, x):
        
        return self.norm.pdf(x)
    

def likelihood(x_, mu1):
    
    
    probs = sts.norm.cdf(mu1 / np.sqrt(2))  #We are still thinking of these as differences so they need to be divided by sqrt(sigma**2 + sigma**2)
    
    if x_[0] < x_[1]:        
        return probs    
    
    else:        
        return 1 - probs
    
   
        
    
    
 

if __name__ == '__main__':
    
    K = 2 # Number of items being ranked
    J = 1 # Number of subjects
    L = 10 #Number of trials (Subjects can contribute more than one trial to the same condition)
    C = 1 #Number of conditions (Number of conditions. Here we assume a within subjects design, but PyThurstonian will automatically handle different design types)

    #Beta parameters
    beta = np.array([[0.0, 0.4]]) 

    data, sim_scale = simulate_data(K, J, L, C, beta = beta, scale = np.array([1.]), seed = 45645346) #Set a seed to recreate the same dataset

    data['Subj'] = pd.factorize(data['Subj'])[0]+1
    myThurst = thurstonian(design_formula = '~0+Condition', data = data, subject_name = "Subj")   
        
    print(data)
    
    
    # accept_reject = True   
    # pyThurst = True
    # grid = True
    
       
    
    # ##########################
    # #Accept reject algorithm #
    
    # if accept_reject:
    #     N_particles = 1000
    #     accepted_particles = np.empty((N_particles, K))
    #     prior = normal_prior(0, K*3)
                
    #     i = 0        
        
    #     while i < N_particles:    
        
    #         particle = prior.rvs()
    #         y_rep = model(particle)
            
    #         if np.all(y_rep == y_obs):
                
    #             accepted_particles[i] = particle            
    #             i += 1                
    #             print(i)
            
    #     plt.hist(accepted_particles[:,1], normed = True, color = 'b', alpha = 0.25)
    
    
    # # plt.figure()
    # # sns.kdeplot(post_s, label = 'grid_approx')
    # # sns.kdeplot(la['beta'][:,0,1], label = 'NUTS-MCMC')
    # # sns.kdeplot(accepted_particles[:,1], label = 'AR_algor')
    # # sns.despine()
