import numpy as np 
import scipy.stats as sts 

import matplotlib.pyplot as plt 
import pdb
import sys
import pandas as pd
rootpath = 'M:\Transport_Studies\Work_Projects\PSI\Publication_Projects\PyThurstonian'
sys.path.append(rootpath)

from PyThurstonian import ranking as rk 



def analytic_prior(mu, sigma):
    """Return prior log probability"""

    mu_prior = sts.norm.logpdf(mu, 0, 6) #normal prior on mu
    sigma_prior = sts.lognorm.logpdf(sigma, s = 0.5, scale = np.exp(0)) #lognormal prior of sigma

    return mu_prior + sigma_prior

def analytic_likelihood(y, mu, sigma):

    log_lik = np.empty(y.shape[0])

    for i in range(y.shape[0]):

        if (y[i, 0] == 1):
            log_lik[i] = sts.norm.logcdf(0, 0 - mu, np.sqrt(sigma**2 * 2))
        else:
            log_lik[i] = np.log(1 - sts.norm.cdf(0, 0 - mu, np.sqrt(sigma**2 * 2)))

    return log_lik.sum()

def analytic_posterior(y, mu, sigma):
    """Get the log unnormalised posterior"""

    return analytic_prior(mu, sigma) + analytic_likelihood(y, mu, sigma)


def analytic_proposal_distribution(theta, cov = 1.0):

    out = sts.multivariate_normal.rvs(theta, cov = cov)

    out[1] = out[1]

    return out
    


def acceptance_analytic(y, prop_theta, theta):

    p_new = analytic_posterior(y, prop_theta[0], prop_theta[1])
    p_old = analytic_posterior(y, theta[0], theta[1])
   
    return  np.min([1, np.exp(p_new - p_old)])



def mcmc_analytic(N_samples, warmup, init_theta, y):


    all_theta = []

    theta = init_theta 

    t = 0
    i = 0
    while (i < N_samples):
        t += 1
        if (i %50) == 0:
            print("sample: {}".format(i))

        prop_theta = analytic_proposal_distribution(theta, [3.0,1]) #Propose theta

        if prop_theta[1] < 0:
            continue

        accept_p = acceptance_analytic(y, prop_theta, theta)

        if np.random.rand() < accept_p:

            theta = prop_theta
            all_theta.append(theta)
            i += 1

    all_theta = np.array(all_theta)

    return all_theta, i/t
    

if __name__ == '__main__':


    data = pd.read_csv("misc/rank_data.csv")

    true_samps = pd.read_csv("misc/HMC_samples.csv")

   
    Y = data[['Y1', 'Y2']].values
    

 
    init_theta = np.array([0.5, 1.0])   
    #Metropolis steps
    N_samples = 10000
    warmup = 1000

    
    fit1, acc_p_1 = mcmc_analytic(N_samples, warmup, init_theta, Y)    

    print(fit1.mean(0))
    print(fit1.std(0))

    plt.figure()
    plt.plot(fit1[warmup:,0], 'r', alpha = 0.5)   

    plt.figure()
    plt.hist(fit1[warmup:,0], density=True, alpha = 0.2, color = 'r', bins = 20)
    plt.hist(true_samps['beta_zero[1,1]'], density=True, alpha = 0.2, color = 'b', bins = 20)
    

    plt.figure()
    plt.hist(fit1[warmup:,1], density=True, alpha = 0.2, color = 'r', bins = 20)
    plt.hist(true_samps['sigma[1]'], density=True, alpha = 0.2, color = 'b', bins = 20)
    
    
    plt.show()