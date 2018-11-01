import numpy as np 
import scipy.stats as sts 

import matplotlib.pyplot as plt 
import pdb
import sys
rootpath = 'M:\Transport_Studies\Work_Projects\PSI\Publication_Projects\PyThurstonian'
sys.path.append(rootpath)

from PyThurstonian import ranking as rk 

def analytic_prior(mu, sigma):
    """Return prior log probability"""

    mu_prior = sts.norm.logpdf(mu, 0, 6) #normal prior on mu
    sigma_prior = sts.lognorm.logpdf(sigma, 0.5, loc = 0) #lognormal prior of sigma

    return mu_prior + sigma_prior

def analytic_likelihood(y, mu, sigma):

    if (y[0] == 1):
        return sts.norm.logcdf(0, (0 - mu), np.sqrt(sigma**2 * 2))
    else:
        return np.log(1 - sts.norm.cdf(0, (0 - mu), np.sqrt(sigma**2 * 2)))

def analytic_posterior(y, mu, sigma):
    """Get the log unnormalised posterior"""

    return analytic_prior(mu, sigma) + analytic_likelihood(y, mu, sigma)


def censor_prior(mu, sigma, z):

    mu_prior = sts.norm.logpdf(mu, 0, 6) #normal prior on mu
    sigma_prior = sts.lognorm.logpdf(sigma, 0.5, loc = 0) #lognormal prior of sigma
    z_prior = np.sum(sts.norm.logpdf(z, np.array([0, mu]), sigma))

    return mu_prior + sigma_prior + z_prior

def prior_rvs():

    mu_prior = sts.norm.rvs(0, 6) #normal prior on mu
    sigma_prior = sts.lognorm.rvs(0.5, loc = 0) #lognormal prior of sigma
    z_prior = sts.norm.rvs(np.array([0, mu_prior]), sigma_prior)


    return np.concatenate((np.array([mu_prior, sigma_prior]), z_prior))

def sim_likelihood(y, mu, sigma):
    """Calculate the likelihood by simulation"""
    match = 0
    N = 100000

    for i in range(N): 
        y_rep = generate_data(np.array([0,mu]), sigma)

        if np.all(y_rep == y):
            match += 1

    return match / N

def analytic_proposal_distribution(theta, cov = 1.0):

    out = sts.multivariate_normal.rvs(theta, cov = cov)

    out[1] = np.abs(out[1])

    return out

#Generate a dataset
def generate_data(mu, sigma):

    z = sts.norm.rvs(mu, sigma)
    y = rk.rank(z)

    return y


def acceptance_analytic(y, prop_theta, theta):

    p_new = analytic_posterior(y, prop_theta[0], prop_theta[1])
    p_old = analytic_posterior(y, theta[0], theta[1])
   
    return  np.min([1, np.exp(p_new - p_old)])



def mcmc_analytic(N_samples, warmup, init_theta, y):


    all_theta = []

    theta = init_theta 

    i = 0
    while (i < N_samples):

        if (i %50) == 0:
            print("sample: {}".format(i))

        prop_theta = analytic_proposal_distribution(theta) #Propose theta

        accept_p = acceptance_analytic(y, prop_theta, theta)

        if np.random.rand() < accept_p:

            theta = prop_theta
            all_theta.append(theta)
            i += 1

    all_theta = np.array(all_theta)

    return all_theta

def check_rank_match(y, z):

    return np.all(rk.rank(z) == y)

def mcmc_censor(N_samples, y):

    #Initialise 
    match = False

    while not match:
        theta_censor = prior_rvs()
        match = check_rank_match(y, theta_censor[2:])

    all_theta = []    

    i = 0
    while i < N_samples:
        if (i %50) == 0:
            print("sample: {}".format(i))
        prop_theta_censor = analytic_proposal_distribution(theta_censor)
        match = check_rank_match(y, prop_theta_censor[2:])

        if not match:
            continue

        else:

            p_old = censor_prior(theta_censor[0], theta_censor[1], theta_censor[2:])
            p_new = censor_prior(prop_theta_censor[0], prop_theta_censor[1], prop_theta_censor[2:])

            accept_p = np.min([1, np.exp(p_new - p_old)])

            if np.random.rand() < accept_p:

                theta_censor = prop_theta_censor
                all_theta.append(theta_censor)
                i += 1
   
    return np.array(all_theta)

if __name__ == '__main__':

    #Generate
    mu = np.array([0, 0.5])
    sigma = 1.0
    y = generate_data(mu, sigma)


    init_theta = np.array([0.5, 1.0])


    


    

    # init_theta_censor = np.array([2.0, 1.0, ])
    # # prop_theta = analytic_proposal_distribution(init_theta)
    
    # prop_theta =np.array([2.0, 1.0])
    # p_accept = acceptance_analytic(y, init_theta, prop_theta)

    # print(np.exp(analytic_posterior(y, init_theta[0], init_theta[1])))
    # print(np.exp(analytic_posterior(y, prop_theta[0], prop_theta[1])))

    # print(p_accept)
    # p_accept_censor = []

    # while len(p_accept_censor) < 10000:

    #     p = accept_censor(y, init_theta, prop_theta)
              
    #     p_accept_censor.append(p) 

    
    
    # print(np.mean(p_accept_censor))



    #Metropolis steps
    N_samples = 10000
    warmup = 5000    


    fit1 = mcmc_analytic(N_samples, warmup, init_theta, y)
    
    fit2 = mcmc_censor(N_samples, y)


    plt.figure()
    plt.hist(fit1[warmup:,0], density=True, alpha = 0.5, color = 'r')
    plt.hist(fit2[warmup:,0], density=True, alpha = 0.5, color = 'b')
    plt.figure()
    plt.hist(fit1[warmup:,1], density=True, alpha = 0.5, color = 'r')
    plt.hist(fit2[warmup:,1], density=True, alpha = 0.5, color = 'b')


    plt.figure()
    plt.plot(fit1[warmup:,0], color = 'r')
    plt.plot(fit2[warmup:,0], color = 'b')
    plt.show()



