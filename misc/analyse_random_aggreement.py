# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 13:04:57 2018

@author: pscog
"""



#import sys
#import os

#sys.path.append(os.path.dirname(os.getcwd()))

from multiprocessing import Process

import PyThurstonian
from PyThurstonian import thurstonian, hdi

import pandas as pd
import seaborn as sns   
import matplotlib.pyplot as plt
import numpy as np
from simulate_data import simulate_data
import ranking as rk

import pdb
    
if __name__ == '__main__':    
        

    #Set plotting preferences
    sns.set_context("paper")    
    sns.set_style("whitegrid")
    
    output_folder = "article/Figures/"
    
    #Load and preprocess data  
    K = 3 #Number of items
    J = 5 #Number of participants
    L = 1 #Number of trials
    C = 2 #Number of conditions
    
    beta = np.array([[0.0, 1.0, 2.0],
                     [0.0, 0.0, 0.0]])        
    
    data, sim_scale = simulate_data(K, J, L, C, beta = beta)

    data['Subj'] = pd.factorize(data['Subj'])[0]+1
    myThurst = thurstonian(design_formula = '~0+Condition', data = data, subject_name = "Subj")   
        
    
    #Note: When running the sampler use a cmdline console, not iPython from Spyder. Unfortunately this fails due to some quirks in Spyder
    resample = True
    if resample:   
                       
        #Prepare for sampling using multiple cores
        myThurst.pre_sample()        
        
        #Sample with multiprocessing
        P = [Process(target=PyThurstonian.run_sample, args = (5000,), kwargs = {'temp_fnum': i})  for i in range(4)]     
           
        for p in P:
            p.start()            
        for p in P:
            p.join()
            
        #Post process the samples. This must be called
        myThurst.post_sample()
        

        myThurst.save_samples('MCMC/random_samples') #Save the samples for later        
    else:    
        #Load old samples
        myThurst.load_samples('MCMC/random_samples')         
    
       
#    ############################################
#    #------------------------------------------#
#    #--------------Plot model params-----------#
#    #------------------------------------------#
#    #------------------------------------------#
#    ############################################
#    
#    sigma = 1 / myThurst.scale
#    scale_posterior_means = sigma.mean(0)
#    scale_posterior_hdi = hdi(sigma)
#    
#    mid_point = (scale_posterior_hdi.T.sum(0) / 2)    
#    bounds = np.abs(scale_posterior_hdi.T - mid_point).T
#
#    plt.figure()
#    plt.errorbar(x =range(J), y = mid_point, yerr = bounds.T, fmt = 'none', color = 'k')
#    plt.plot([str(i) for i in range(J)], scale_posterior_means, 'ok')
#    plt.plot([str(i) for i in range(J)], 1/sim_scale, 'or')
#    
#    plt.xlabel('Participant')
#    plt.ylabel('Scale')
    
    ############################################
    #------------------------------------------#
    #--------------Plot Agreement--------------#
    #------------------------------------------#
    #------------------------------------------#
    ############################################
#    all_tau = myThurst.pairwise_tau_distance(level_dict_1 = {'Condition':'C1'}, 
#                                   level_dict_2 = {'Condition':'C2'})            
    
    average_pairwise_tau = myThurst.average_pairwise_tau_distance(level_dict_1 = {'Condition':'C1'}, 
                                   level_dict_2 = {'Condition':'C2'}) 
    
    fig, ax = plt.subplots(1, 2, figsize = (4, 2), sharex = True, sharey = True)
    myThurst.plot_sra(level_dict = {'Condition':  'C1'}, ax = ax[0])
    myThurst.plot_sra(level_dict = {'Condition':  'C2'}, ax = ax[1])
    
    plt.suptitle("Bayesian Agreement")  
    [ax[i].set_xlabel("SRA") for i in range(2)]
    plt.savefig(output_folder + "simple_agg2.png")
    
    ############################################
    #------------------------------------------#
    #------------Plot the raw data-------------#
    #------------------------------------------#
    #------------------------------------------#
    ############################################
    
    conds = ['C1', 'C2']    
    fig, ax = plt.subplots(1,2, sharey = True, figsize = (3, 2))
    
    #Iterate over every condition
    for i, cond in enumerate(conds):              
 
        myThurst.plot_data(level_dict = {'Condition': cond}, ax = ax[i])
        ax[i].set_title("Condition: {}".format(cond))
        ax[i].set_ylim([0,1])
        
    plt.suptitle("Data Summary")
    
    plt.subplots_adjust(top=0.76,
                        bottom=0.34,
                        left=0.185,
                        right=0.952,
                        hspace=0.2,
                        wspace=0.327)
 
    plt.savefig(output_folder + "simple_raw_data.png")
    
    ############################################
    #------------------------------------------#
    #-------Plot the Bayesian Aggregates-------#
    #------------------------------------------#
    #------------------------------------------#
    ############################################
    
    fig, ax = plt.subplots(1,2, sharey = True, figsize = (3, 2))
    
    #Iterate over every condition
    for i, cond in enumerate(conds):                
 
            myThurst.plot_aggregate(level_dict = {'Condition':cond}, ax = ax[i])            
            ax[i].set_title("Condition: {}".format(cond))
    
    ax[0].set_ylim([0, 1])
    plt.suptitle("Bayesian Aggregate")
    
    plt.subplots_adjust(top=0.765,
                        bottom=0.34,
                        left=0.184,
                        right=0.952,
                        hspace=0.2,
                        wspace=0.322)
    
    
    plt.savefig(output_folder + "simple_agg.png")
    
    ############################################
    #------------------------------------------#
    #-------Plot the Bayesian Contrasts--------#
    #------------------------------------------#
    #------------------------------------------#
    ############################################
    
    fig, ax = plt.subplots(1,1, sharey = True, figsize = (2,1.5))              
 
    myThurst.plot_contrast(level_dict_1 = {'Condition':'C1'}, 
                           level_dict_2 = {'Condition':'C2'},
                           ax = ax)            
    
    average_pairwise_tau = myThurst.average_pairwise_tau_distance(level_dict_1 = {'Condition':'C1'}, 
                                   level_dict_2 = {'Condition':'C2'}) 
    
    print('pariwise = {:2.2f}'.format(average_pairwise_tau))
    
    ax.set_ylim([0, 1])
    ax.set_title("Contrast: C1 vs C2")
    plt.tight_layout()
    
    plt.savefig(output_folder + "simple_contrast.png")
    
    
    plt.show()