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

import ranking as rk

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
    
    
    data = pd.read_csv("raw_data/VET_data/VET_data.csv".format(1))    
    data = pd.pivot_table(data, values = "Rating", index = ["ID", "Question", "Q_n", "Condition", "Base"], columns = "Height").reset_index()
      
#    data = data.rename(columns = {'H': 'Y1', 'M': 'Y2', 'L': 'Y3'}) #Rename the columns
    
    data[['Y1', 'Y2', 'Y3']] = rk.rank(data[['H', 'M', 'L']])
    
    q_data = data[data['Q_n'] == 1]
    q_data = q_data[q_data['Base'] != 'JLR']
    
    q_data['ID'] = q_data['ID'] + 1
    
    myThurst = thurstonian(design_formula = '~Base', data = q_data, subject_name = "ID")        
    
    resample = False
    
    if resample:   
        
        #NOTE: IF using multiprocessing you will need to run in an external cmd window. Will not work in Spyder
        #Prepare for sampling using multiple cores
        myThurst.pre_sample()      
        
    
        
        #Sample with multiprocessing
        P = [Process(target=PyThurstonian.run_sample, args = (5000,), kwargs = {'temp_fnum': i})  for i in range(4)]     
           
        for p in P:
            p.start()            
        for p in P:
            p.join()
            
#        Post process the samples. This must be called
        myThurst.post_sample()
        myThurst.save_samples('MCMC/VET_samples') #Save the samples for later
        
    else:    
        #Load old samples
        myThurst.load_samples('MCMC/VET_samples')
    
        
#    #The factors and their levels
#    tracks = ['CLV', 'LR2']
    bases = ['Real', '3m', '6m', 'FM']    
#    
#    ############################################
#    #------------------------------------------#
#    #------------Plot the raw data-------------#
#    #------------------------------------------#
#    #------------------------------------------#
#    ############################################
    
    fig, ax = plt.subplots(1, 4, sharey = True, figsize = (6.2, 2.0))   
    #Iterate over every condition
    
    for j, base in enumerate(bases):               
 
        myThurst.plot_data(level_dict = {'Base': base}, ax = ax[j])
        ax[j].set_title( "Base: {}".format(base))

    plt.suptitle("Data Summary")

    plt.subplots_adjust(top=0.776,
                        bottom=0.301,
                        left=0.075,
                        right=0.977,
                        hspace=0.2,
                        wspace=0.286)    
    
    plt.savefig(output_folder + "VET_raw_data.png")
    
#    ############################################
#    #------------------------------------------#
#    #-------Plot the Bayesian Aggregates-------#
#    #------------------------------------------#
#    #------------------------------------------#
#    ############################################
    
    fig, ax = plt.subplots(1, 4, sharey = True, figsize = (6.2, 2.0))
    
    #Iterate over every condition
   
    for j, base in enumerate(bases):               
 
        myThurst.plot_aggregate(level_dict = {'Base': base}, ax = ax[j])
        ax[j].set_title("Base: {}".format( base))

    plt.suptitle("Bayesian Aggregate")
  
    plt.subplots_adjust(top=0.776,
                        bottom=0.301,
                        left=0.095,
                        right=0.977,
                        hspace=0.2,
                        wspace=0.286)   
        
    plt.savefig(output_folder + "VET_agg.png")
    
#    ############################################
#    #------------------------------------------#
#    #-------Plot the Bayesian Contrasts--------#
#    #------------------------------------------#
#    #------------------------------------------#
#    ############################################
    
    fig, ax = plt.subplots(1, 3, sharey = True, figsize = (5,1.7))
    
    #Iterate over every condition
    
    for j, base in enumerate(bases[1:]):               
 
        myThurst.plot_contrast(level_dict_1 = {'Base': 'Real'}, 
                               level_dict_2 = {'Base': base},
                               ax = ax[j])            
        
        ax[j].set_title("Base: Real vs {}".format(base))

    plt.suptitle("Contrasts")
    
    plt.subplots_adjust(top=0.758,
                        bottom=0.244,
                        left=0.119,
                        right=0.971,
                        hspace=0.2,
                        wspace=0.305)
    
    plt.savefig(output_folder + "VET_contrasts.png")
    
#    
#    ############################################
#    #------------------------------------------#
#    #--------------Plot Agreement--------------#
#    #------------------------------------------#
#    #------------------------------------------#
#    ############################################
#    all_tau = myThurst.pairwise_tau_distance(level_dict_1 = {'Condition':'C1'}, 
#                                   level_dict_2 = {'Condition':'C2'})            
    
#    average_pairwise_tau = myThurst.average_pairwise_tau_distance(level_dict_1 = {'Condition':'C1'}, 
#                                   level_dict_2 = {'Condition':'C2'}) 
    
    fig, ax = plt.subplots(1, 4, sharey = True, figsize = (6.5,2.2))
    
    #Iterate over every condition
  
    for j, base in enumerate(bases): 
        myThurst.plot_sra(level_dict = {'Base': base}, ax = ax[j])
    
    
    plt.suptitle("Bayesian Agreement")  
    [ax[j].set_xlabel("SRA") for i in range(4)]
    
    
    plt.subplots_adjust(top=0.844,
                        bottom=0.228,
                        left=0.044,
                        right=0.978,
                        hspace=0.2,
                        wspace=0.123)
        
    plt.savefig(output_folder + "VET_aggreement.png")
    
    plt.show()
    
    