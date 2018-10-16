# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 13:04:57 2018

@author: pscog
"""

from multiprocessing import Process

import PyThurstonian
from PyThurstonian import thurstonian, hdi

import pandas as pd
import seaborn as sns   
import matplotlib.pyplot as plt

if __name__ == '__main__':    
    
    output_folder = "article/Figures/"
    #Set plotting preferences
    sns.set_context("paper")    
    sns.set_style("whitegrid")
    
    ############################################
    #------------------------------------------#
    #--------Load the data and process---------#
    #------------------------------------------#
    #------------------------------------------#
    ############################################
    
    
    data = pd.read_csv("raw_data/scs_data/Q{}.csv".format(1))          
    myThurst = thurstonian(design_formula = '~Test*Base', data = data, subject_name = "Driver")        
    
    resample = False
    
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
        myThurst.save_samples('MCMC/SCS_samples') #Save the samples for later
        
    else:    
        #Load old samples
        myThurst.load_samples('MCMC/SCS_samples')
    
        
    #The factors and their levels
    tracks = ['CLV', 'LR2']
    bases = ['Real', '3M', '6M', 'FM']    
    
    ############################################
    #------------------------------------------#
    #------------Plot the raw data-------------#
    #------------------------------------------#
    #------------------------------------------#
    ############################################
    
    fig, ax = plt.subplots(2, 4, sharey = True, figsize = (8,3.8))   
    #Iterate over every condition
    for i, tr in enumerate(tracks):
        for j, base in enumerate(bases):               
 
            myThurst.plot_data(level_dict = {'Test':tr, 'Base': base}, ax = ax[i,j])
            ax[i,j].set_title("Track: {}, Base: {}".format(tr, base))

    plt.suptitle("Data Summary")
    
    plt.subplots_adjust(top=0.868,
                        bottom=0.152,
                        left=0.082,
                        right=0.982,
                        hspace=0.931,
                        wspace=0.212)    
    
    plt.savefig(output_folder + "scs_raw_data.png")
    
    ############################################
    #------------------------------------------#
    #-------Plot the Bayesian Aggregates-------#
    #------------------------------------------#
    #------------------------------------------#
    ############################################
    
    fig, ax = plt.subplots(2, 4, sharey = True, figsize = (8,5))
    
    #Iterate over every condition
    for i, tr in enumerate(tracks):
        for j, base in enumerate(bases):               
 
            myThurst.plot_aggregate(level_dict = {'Test':tr, 'Base': base}, ax = ax[i,j])
            ax[i,j].set_title("Track: {}, Base: {}".format(tr, base))
    
    plt.suptitle("Bayesian Aggregate")
    
    plt.subplots_adjust(top=0.886,
                        bottom=0.135,
                        left=0.069,
                        right=0.982,
                        hspace=0.538,
                        wspace=0.218)
        
    plt.savefig(output_folder + "scs_agg.png")
    
    ############################################
    #------------------------------------------#
    #-------Plot the Bayesian Contrasts--------#
    #------------------------------------------#
    #------------------------------------------#
    ############################################
    
    fig, ax = plt.subplots(2, 3, sharey = True, figsize = (6,4))
    
    #Iterate over every condition
    for i, tr in enumerate(tracks):
        for j, base in enumerate(bases[1:]):               
 
            myThurst.plot_contrast(level_dict_1 = {'Test':tr, 'Base': 'Real'}, 
                                   level_dict_2 = {'Test':tr, 'Base': base},
                                   ax = ax[i,j])            
            
            ax[i,j].set_title("Track: {},\nBase: Real vs {}".format(tr, base))
    
    plt.suptitle("Contrasts")
    
    plt.subplots_adjust(top=0.836,
                        bottom=0.128,
                        left=0.099,
                        right=0.976,
                        hspace=0.746,
                        wspace=0.238)
    
    plt.savefig(output_folder + "scs_contrasts.png")
    
    
    ############################################
    #------------------------------------------#
    #--------------Plot Agreement--------------#
    #------------------------------------------#
    #------------------------------------------#
    ############################################
#    all_tau = myThurst.pairwise_tau_distance(level_dict_1 = {'Condition':'C1'}, 
#                                   level_dict_2 = {'Condition':'C2'})            
    
#    average_pairwise_tau = myThurst.average_pairwise_tau_distance(level_dict_1 = {'Condition':'C1'}, 
#                                   level_dict_2 = {'Condition':'C2'}) 
    
    fig, ax = plt.subplots(2, 4, sharey = True, figsize = (8,5))
    
    #Iterate over every condition
    for i, tr in enumerate(tracks):
        for j, base in enumerate(bases): 
            myThurst.plot_sra(level_dict = {'Test':tr, 'Base': base}, ax = ax[i, j])
    
    
    plt.suptitle("Bayesian Agreement")  
    [ax[i, 0].set_xlabel("SRA") for i in range(2)]
    plt.savefig(output_folder + "scs_aggreement.png")
    
    
    plt.show()
    
    