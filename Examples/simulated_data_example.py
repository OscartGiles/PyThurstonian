"""An example of using PyThurstonian using a simualted data set"""

#This is just for working with PyThurstonian on my machine. Should remove on release
import sys

from PyThurstonian import thurstonian, simulate_data, run_sample, hdi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from multiprocessing import Process #If we want to use multiprocessing

import pdb
import seaborn as sns

if __name__ == '__main__': #PyThurstonian can use the Python multiprocessing library. To use it all code must be within main

    sns.set(font_scale=0.8)
    ############################################
    #------------------------------------------#
    #---------Create a fake data set-----------#
    #------------------------------------------#
    #------------------------------------------#
    ############################################

    ##Dataset properties
    K = 3 # Number of items being ranked
    J = 10 # Number of subjects
    L = 1 #Number of trials (Subjects can contribute more than one trial to the same condition)
    C = 2 #Number of conditions (Number of conditions. Here we assume a within subjects design, but PyThurstonian will automatically handle different design types)

    #Beta parameters
    beta = np.array([[0.0, 1.0, 2.0],
                    [0.0, 0.1, 0.2]]) 


    data, sim_scale = simulate_data(K, J, L, C, beta = beta, seed = 543235) #Set a seed to recreate the same dataset

    data.to_csv("sim_rank_data.csv")
   
    ############################################
    #------------------------------------------#
    #-------------Define the BTM---------------#
    #------------------------------------------#
    #------------------------------------------#
    ############################################

    data['Subj'] = pd.factorize(data['Subj'])[0]+1
    myThurst = thurstonian(design_formula = '~0+Condition', data = data, subject_name = "Subj")   
    
 
    cores = 6 #Set the number of cores you want to use on your machine
    resample = True #Set to true to sample from the posterior and save the fit for later. If False will load previous samples

    if not resample:
        myThurst.load_samples('Examples/MCMC/simple_samples')  #Pass the location you want to save the samples (make sure the directory exists)

    else:

        if cores == 1:
            #We can use the following function to fit on a singel core
            myThurst.sample( iter = 5000, warmup = None, adapt_delta = 0.99, priors = None, rep_data = True, refresh = 250, write_sample_file = None)

        else:

            #We can run multiple cores using pythons multiprocessing module

            #Call this function to prepare everything to run on multiple cores
            myThurst.pre_sample()        

            #Sample with multiprocessing
            P = [Process(target=run_sample, args = (5000,), kwargs = {'temp_fnum': i})  for i in range(cores)]     
           
            #Start sampling and wait for everything to join
            for p in P:
                p.start()            
            for p in P:
                p.join()
            
            #This must be called after sampling - PyThurstonian will then gather all the results
            myThurst.post_sample()
        

        # myThurst.save_samples('Examples/MCMC/simple_samples') #Save the samples for later

    

    p1 = myThurst.predictive_prob_agreement({'Condition':'C1'})
    p2 = myThurst.predictive_prob_agreement({'Condition':'C2'})

    print(p1, p2)
    ############################################
    #------------------------------------------#
    #-------------Plot the results-------------#
    #------------------------------------------#
    #------------------------------------------#
    ############################################

    SAVE_FIGS = True
    output_folder = './article/Figures/'

    ############################################
    #------------------------------------------#
    #--------------Plot model params-----------#
    #------------------------------------------#
    #------------------------------------------#
    ############################################
    
    sigma = 1 / myThurst.scale
    scale_posterior_means = sigma.mean(0)
    scale_posterior_hdi = hdi(sigma)
    
    mid_point = (scale_posterior_hdi.T.sum(0) / 2)    
    bounds = np.abs(scale_posterior_hdi.T - mid_point).T

    plt.figure()
    plt.errorbar(x =range(J), y = mid_point, yerr = bounds.T, fmt = 'none', color = 'k')
    plt.plot([str(i) for i in range(J)], scale_posterior_means, 'ok')
    plt.plot([str(i) for i in range(J)], 1/sim_scale, 'or')
    
    plt.xlabel('Participant')
    plt.ylabel('Scale')
    
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
    
    fig, ax = plt.subplots(1, 2, figsize = (3.5, 2), sharex = True, sharey = True)
    # myThurst.plot_sra(level_dict = {'Condition':  'C1'}, ax = ax[0])
    # myThurst.plot_sra(level_dict = {'Condition':  'C2'}, ax = ax[1])
    myThurst.plot_kendall_W(level_dict = {'Condition':  'C1'}, ax = ax[0])
    myThurst.plot_kendall_W(level_dict = {'Condition':  'C2'}, ax = ax[1], text_x_pos=.4)
    
    ax[0].set_title("Condition: C1")
    ax[1].set_title("Condition: C2")
    plt.suptitle("Bayesian Agreement")  

    plt.tight_layout()
    plt.subplots_adjust(top=0.765,
                    bottom=0.191,
                    left=0.121,
                    right=0.961,
                    hspace=0.2,
                    wspace=0.222)
                    

    if SAVE_FIGS:
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
    
    if SAVE_FIGS:
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
    
    plt.suptitle("Bayesian Aggregate")
    
    plt.subplots_adjust(top=0.765,
                        bottom=0.34,
                        left=0.184,
                        right=0.952,
                        hspace=0.2,
                        wspace=0.322)
    
    if SAVE_FIGS:
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
    
    if SAVE_FIGS:
        plt.savefig(output_folder + "simple_contrast.png")
    
    
    plt.show()