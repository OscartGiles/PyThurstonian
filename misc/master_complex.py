# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 13:04:57 2018

@author: pscog
"""

import pandas as pd
import numpy as np
import patsy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle 

import pystan 
import ranking as rk

import seaborn as sns

import pdb

def get_cond_idx(conds):
    return dict(zip(conds, range(len(conds))))


def plot_raw_data_complex(conds, cond_plot, stan_data, ax = None, color = '#E4F1FE'):
    """Plot the propotion of subjects who chose each possible ranking"""
    
    if ax == None:
        
        ax = plt.gca()
        
    cond_idx = get_cond_idx(conds)
    N_conds = stan_data['C']
    
    X_rep = np.diag(np.ones(N_conds))
            
    cond_data = y[np.all(X == X_rep[cond_idx[cond_plot]], axis = -1)]        
    
    bord_count = rk.borda_count(cond_data)
    
#    bord_count_idx = bord_count - 1

    tdist = rk.kendall_tau_dist_vec(np.tile(bord_count, (cond_data.shape[0],1)), cond_data)
    
    ax.hist(tdist, normed = True, rwidth = 0.8, color = '#FFC0CB')
    
            
    return 

def plot_bayesian_aggregate_complex(mean_rank, all_ranks, conds, cond_plot, ax = None):
    
    if ax == None:
        
        ax = plt.gca()

    cond_idx = get_cond_idx(conds)
    
    est_True_mean = mean_rank[cond_idx[cond_plot]]
    
    tau_dist = rk.kendall_tau_dist_vec(np.tile(est_True_mean, (all_ranks.shape[0],1)), all_ranks[:, cond_idx[cond_plot]])
    
    tau_bins = rk.bin_taus(tau_dist, all_ranks.shape[-1])
    tau_bins_0 = [float(i) for i in tau_bins[0]]
    width = np.diff(tau_bins_0).min()
    
    colors = np.repeat('#E4F1FE', len(tau_bins[0]))
                 
    
                       
    ax.bar(tau_bins_0, tau_bins[1] / tau_bins[1].sum(), width, color = colors, edgecolor = ['k' for i in range(len(tau_bins[0]))])
  
#    plt.xticks(np.arange(0, 1+0.25, 0.25))
#    ax.hist(tau_dist, normed = True, rwidth = 1, color = '#FFC0CB', alpha = 0.8, cumulative = False)
#    ax.hist(tau_dist, normed = True, rwidth = 1, color = '#22313F', alpha = 0.8, cumulative = True)
                       
  
    
    return est_True_mean, tau_dist


def plot_bayesian_contrast(all_ranks, conds, cond1, cond2, ax = None):
    
    if ax == None:
        
        ax = plt.gca()

    cond_idx = get_cond_idx(conds)
    
    comp_taus = rk.kendall_tau_dist_vec(all_ranks[:,cond_idx[cond1]], all_ranks[:,cond_idx[cond2]])
    
    comp_bins = rk.bin_taus(comp_taus, K)    
    
    colors = np.repeat('#E4F1FE', len(comp_bins[0]))   
                    
    ax.bar(comp_bins[0], comp_bins[1] / comp_bins[1].sum(),
                  color = colors, edgecolor = ['k' for i in range(len(comp_bins[0]))]) 
    
    
def plot_bayesian_agreement(all_ranks, conds, cond, ax = None):
    
    if ax == None:
        
        ax = plt.gca()
    cond_idx = get_cond_idx(conds)
    X_rep = np.diag(np.ones(N_conds))
    
    X_rep == X_rep[cond_idx[cond]]
    
    cond_mask = np.all(X == X_rep[cond_idx[cond]], axis = -1)   
                
    z_rep_cond = z_rank[:, cond_mask]    
    
    all_z_tau = np.empty(z_rep_cond.shape[:2])    
    
    for part in range(z_rep_cond.shape[1]):
        all_z_tau[:, part] = rk.kendall_tau_dist_vec(z_rep_cond[:,part], all_ranks[:, cond_idx[cond]])

    ax.hist(all_z_tau.mean(1), color = '#E4F1FE', rwidth = 0.85, normed = True)
            
            
            
    
if __name__ == '__main__':    
        
    #Set plotting preferences
    sns.set_context("paper")    
    sns.set_style("whitegrid")
    
    #Load and preprocess data
    data = pd.read_csv("sim_data_complex.csv")   
    
    conds = data['Condition'].unique()    
    N_conds = len(conds)
    X = patsy.dmatrix("~0+Condition", data = data)    
    y = data[[i for i in data.columns if 'Y' in i]].values
    
    part = pd.factorize(data['Subject'])
    
    N = y.shape[0] #Number of data points
    K = y.shape[1] #Number of items to rank
    C = X.shape[1] #Number of predictors (or number of conditions)
    J = np.unique(part[0]).shape[0] #Number of participants    
    
    stan_data = {'N': N, 'K': K, 'C': C, 'J': J, 'X': X, 'y': y, 'rater': part[0] + 1, 'beta_sd_prior': C*3, 
                 'scale_sd_prior': 0.5, 'rep_data': 1} #Stan data dictionary
      
    
    resample = False
    ##Fit the stan model (This could be done in RStan or any other stan interface is required)
    try: 
        with open("thurstonian.pkl", 'rb') as f:
            sm = pickle.load(f)
    except:
        sm = pystan.StanModel(file="thurstonian_cov.stan")
        with open("thurstonian.pkl", 'wb') as f:
            pickle.dump(sm, f)
            
    
    if resample:        
        fit = sm.sampling(data=stan_data, iter=4000, chains=4, n_jobs = 1)
        la = fit.extract(permuted = True)
        with open("la_complex.pkl", 'wb') as f:
            pickle.dump(la, f)
        
    else:    
        with open("la_complex.pkl", 'rb') as f:
            la = pickle.load(f)            
            
    #Extract mcmc samples for the parameters of interest
    beta = la['beta']
    z_rank = la['z_rank']
    
    ##Post process and visualization 
    mean_rank = rk.rank(beta.mean(0), axis = -1) #Get the posterior mean rank
    all_ranks = rk.rank(beta, axis = -1) #Rank all the beta samples
        
#    
#    
#    ###Main plots       
#    
#    ##Plot the raw data
    cond_order = ['C1', 'C2', 'C3']
#    
#    fig, ax = plt.subplots(1, len(conds), sharex = True, sharey = True, figsize = (6, 2.5))
#    
#    for i, c in enumerate(cond_order):
#        print(i)
#        plot_raw_data_complex(conds, c, stan_data, ax = ax[i])
#        ax[i].set_xlim([0,1])

        

#        borda_text = "-".join(map(str, borda_count))
#        ax[i].text(0.02, 0.88, "Borda: {}".format("-".join(map(str, borda_count))), transform=ax[i].transAxes)
    
#        ax[i].set_title("Condition = {}".format(c))
#        plt.sca(fig.axes[i])
#        plt.ylim([0,1])
#        plt.xticks(rotation=60)
#        
#        ax[i].set_xlabel("Ranking")
#        ax[i].set_ylabel("Proportion Ranking")
#    
#    plt.tight_layout()
    
#    
#    #Plot the mcmc samples
    sns.set_style("white")
    fig, ax = plt.subplots(1, len(conds), sharex = True, sharey = True, figsize = (6, 2))
    
    for i, c in enumerate(cond_order):
        
        T_mean, tdist = plot_bayesian_aggregate_complex(mean_rank, all_ranks, conds, c, ax = ax[i])  

        ax[i].xaxis.set_major_locator(ticker.MultipleLocator(0.25))
        ax[i].set_xlim([0, 1])
        sns.despine()
        
        confidence_interval = np.sort(tdist)[int(tdist.shape[0] * 0.95)]
        ax[i].fill_between([0, confidence_interval], 0, 0.4, color = 'r', alpha = 0.25)
#        ax[i].errorbar(confidence_interval/2, 0.005, xerr = confidence_interval/2, fmt='none', color = 'k', elinewidth = 2)
        
#        text = "-".join(map(str, T_mean))
##        ax[i].text(0.02, 0.88, "Mean est: {}".format(text), transform=ax[i].transAxes)
##        plt.sca(fig.axes[i])
##        plt.xticks(rotation=60)
##        plt.ylim([0,1])
        ax[i].set_title("Condition = {}".format(c))
        ax[i].set_xlabel("Kendall Tau")
        ax[i].set_ylabel("P(Kendall Tau)")
##        
    plt.tight_layout()
    
    
    sns.set_style("whitegrid")
    #Plot agreement with the aggregate rank
    fig, ax = plt.subplots(1, len(conds), sharex = True, sharey = True, figsize = (6, 2.5))
    
    for i, c in enumerate(cond_order):
        plot_bayesian_agreement(all_ranks, conds, c, ax = ax[i])
        ax[i].set_title("Condition = {}".format(c))
        ax[i].axvline(0.5, linestyle = '--', color = 'k')
        
    ax[0].set_xlim([0, 1])

    plt.tight_layout()
#    #Comparision betweeen conditions
#    fig, ax = plt.subplots(1, 1, sharex= True, sharey = True, figsize = (2.5, 2))
    
#    plot_bayesian_contrast(all_ranks, conds, "C1", "C2", ax = ax)
    
#    plt.ylim([0,1])
#    ax.set_title("Contrast: {} vs {}".format("Real", "Sim1"))
#    #ax[1].set_title("{} vs {}".format("Real", "Sim2"))
#    #ax[2].set_title("{} vs {}".format("Sim1", "Sim2"))
#    
#    ax.set_xlabel("Kendall Tau Distance")
#    ax.set_ylabel("P(Kendal Tau Distance)")
#    plt.tight_layout()
#     #Comparison plots           
#      
                    
     
    