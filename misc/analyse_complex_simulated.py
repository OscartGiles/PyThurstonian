# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 11:26:32 2018

@author: pscog
"""

from PyThurstonian import thurstonian
import pandas as pd
import seaborn as sns   
import matplotlib.pyplot as plt
import numpy as np
from simulate_data import simulate_data
import ranking as rk


if __name__ == '__main__':      
        
    #Set plotting preferences
    sns.set_context("paper")    
    sns.set_style("whitegrid")
    
    #Load and preprocess data  
    K = 15
    J = 20 #Number of participants
    L = 1 #Number of trials
    C = 2 #Number of conditions
    
    beta = [np.arange(0, 15), np.arange(0, 15)]
    beta[1] = beta[1][::-1]

    beta = np.array(beta) *0.1
    
#    beta = np.zeros((C, K))
    
    
    data = simulate_data(K, J, L, C, beta = beta, seed = 234254324)


    data['Subj'] = pd.factorize(data['Subj'])[0]+1
    myThurst = thurstonian(design_formula = '~Condition', data = data, subject_name = "Subj")        
    
    resample = True
    
    if resample:   
        myThurst.sample(iter = 5000, chains = 1) #Sample from the posterior    
        myThurst.save_samples('MCMC/complex_samples') #Save the samples for later
        
    else:    
        #Load old samples
        myThurst.load_samples('MCMC/complex_samples')   
        
    
    ############################################
    #------------------------------------------#
    #-------Plot the Bayesian Aggregates-------#
    #------------------------------------------#
    #------------------------------------------#
    ############################################
            
    conds = ['C1', 'C2']
    
    fig, ax = plt.subplots(1,2, sharey = True, figsize = (3, 2))


    #Iterate over every condition
    for i, cond in enumerate(conds):                
 
            myThurst.plot_aggregate_complex(level_dict = {'Condition':cond}, ax = ax[i])            
            a = myThurst.get_p_distance(level_dict = dict(Condition = cond), alpha = .95)
            
            ax[i].text(0.5, 0.75, "HDI={:3.3f}".format(a), transform = ax[i].transAxes, fontsize = 7)
            ax[i].axvline(a, linestyle = '--', color = 'k')
            ax[i].set_title("Condition: {}".format(cond))
            ax[i].set_xlim([0, 1])
    
    plt.suptitle("Bayesian Aggregate")
    
    plt.subplots_adjust(top=0.795,
                        bottom=0.192,
                        left=0.157,
                        right=0.961,
                        hspace=0.2,
                        wspace=0.397)


    ############################################
    #------------------------------------------#
    #-------Plot the Bayesian Contrasts--------#
    #------------------------------------------#
    #------------------------------------------#
    ############################################
    
    fig, ax = plt.subplots(1,1, sharey = True, figsize = (2,1.5))              
 
    myThurst.plot_contrast_complex(level_dict_1 = {'Condition':'C1'}, 
                           level_dict_2 = {'Condition':'C2'},
                           ax = ax)          
    
    ax.set_xlim([0, 1])
    ax.set_title("Contrast: C1 vs C2")
    plt.tight_layout()
    
#    plt.savefig(output_folder + "simple_contrast.png")
#for i, c in enumerate(cond_order):
#    
#    T_mean, tdist = plot_bayesian_aggregate_complex(mean_rank, all_ranks, conds, c, ax = ax[i])  
#
#    ax[i].xaxis.set_major_locator(ticker.MultipleLocator(0.25))
#    ax[i].set_xlim([0, 1])
#    sns.despine()
#    
#    confidence_interval = np.sort(tdist)[int(tdist.shape[0] * 0.95)]
#    ax[i].fill_between([0, confidence_interval], 0, 0.4, color = 'r', alpha = 0.25)
#        ax[i].errorbar(confidence_interval/2, 0.005, xerr = confidence_interval/2, fmt='none', color = 'k', elinewidth = 2)
    
#        text = "-".join(map(str, T_mean))
##        ax[i].text(0.02, 0.88, "Mean est: {}".format(text), transform=ax[i].transAxes)
##        plt.sca(fig.axes[i])
##        plt.xticks(rotation=60)
##        plt.ylim([0,1])
#    ax[i].set_title("Condition = {}".format(c))
#    ax[i].set_xlabel("Kendall Tau")
#    ax[i].set_ylabel("P(Kendall Tau)")
##        

    
       