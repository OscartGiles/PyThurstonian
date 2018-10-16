# -*- coding: utf-8 -*-
"""
Created on Wed May  2 13:20:00 2018

@author: pscog
"""


from PyThurstonian import thurstonian
import pandas as pd
import seaborn as sns   
import matplotlib.pyplot as plt
import numpy as np
from simulate_data import simulate_data
import ranking as rk

import pystan
import patsy 

import pickle


try:
    gen = pickle.load(open("thurstonian_gen.pkl", 'rb'))
    
except FileNotFoundError:
       
    gen =  pystan.StanModel(file = "thurstonian_cov_generate.stan")
    
    with open('thurstonian_gen.pkl', 'wb') as f:
        
        pickle.dump(gen, f)
        
        
try:
    sm = pickle.load(open("thurstonian.pkl", 'rb'))

except FileNotFoundError:
       
    sm =  pystan.StanModel(file = "thurstonian_cov.stan")
    
    with open('thurstonian_cov.pkl', 'wb') as f:
        
        pickle.dump(sm, f)
        
        
if __name__ == '__main__':    
        
    #Set plotting preferences
    sns.set_context("paper")    
    sns.set_style("whitegrid")
    
    output_folder = "article/Figures/"
    
    #######################################
    #-------------------------------------#
    #---------Data params-----------------#
    #-------------------------------------#
    #-------------------------------------#
    #-------------------------------------#
    ####################################### 
     
    K = 3 #Number of items
    J = 10 #Number of participants
    L = 1 #Number of trials
    C = 2 #Number of conditions    
    
    
    #######################################
    #-------------------------------------#
    #-------------SBC params--------------#
    #-------------------------------------#
    #-------------------------------------#
    #-------------------------------------#
    ####################################### 
    
    N = 5000
    
           
    #######################################
    #-------------------------------------#
    #----------Generate test data---------#
    #-------------------------------------#
    #-------------------------------------#
    #-------------------------------------#
    #######################################    
    
    
    beta = np.array([[0.0, 1.0, 2.0],
                     [0.0, -0.1, -0.2]])          
    data = simulate_data(K, J, L, C, beta = beta, seed = 8547346)
    data['Subj'] = pd.factorize(data['Subj'])[0]+1    
    y = data[['Y1', 'Y2', 'Y3']].values
    X = np.asarray(patsy.dmatrix('~Condition', data = data))
    subj = data['Subj'].values   
    stan_data = dict(N = y.shape[0], K = y.shape[1], C = X.shape[1], J = J, rater = subj, X = X, beta_sd_prior = K * 3, scale_sd_prior = 0.5)


    #Generate all data simulations    
    fit = gen.sampling(data = stan_data, iter = N, warmup = 0, algorithm="Fixed_param", n_jobs = 1, chains = 1)    
    la = fit.extract()

    y_rep = la['y'].astype(int)
    
    rank = np.empty(N)
    
    for i in range(N):
                  
        stan_data = dict(N = y_rep[i].shape[0], K = y_rep[i].shape[1], C = X.shape[1], J = J, rater = subj, X = X, y = y_rep[i], beta_sd_prior = K * 3, scale_sd_prior = 0.5)
    
        fit = sm.sampling(data = stan_data, chains = 1, n_jobs = 1, warmup = 500, iter = 510)
        la_fit = fit.extract()
        
        
        scale_posterior = la_fit['scale'][:,0]    
        scale_prior = la['scale'][i,0]
    
        rank[i] = (scale_prior > scale_posterior).sum()
    
    count = np.array([(rank == i).sum() for i in range(10)])
    
    plt.bar(range(10), count / count.sum() )
#    myThurst = thurstonian(design_formula = '~0+Condition', data = data, subject_name = "Subj")   
#        
#    resample = False
#    if resample:   
#        myThurst.sample(iter = 2000, chains = 4, adapt_delta = 0.9999, n_jobs = 1) #Sample from the posterior    
#        myThurst.save_samples('MCMC/simple_samples') #Save the samples for later        
#    else:    
#        #Load old samples
#        myThurst.load_samples('MCMC/simple_samples')      