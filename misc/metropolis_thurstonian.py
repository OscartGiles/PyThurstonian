import numpy as np 
import scipy.stats as sts 

import matplotlib.pyplot as plt 
import pdb
import sys
rootpath = 'M:\Transport_Studies\Work_Projects\PSI\Publication_Projects\PyThurstonian'
sys.path.append(rootpath)

from PyThurstonian import ranking as rk 


def ordered_vector(X):
    """Transform an ordered vector to unconstrained space"""

    Y = np.empty(X.size)

    for i in range(X.size):

        if i == 0:
            Y[i] = X[i]

        else:
            Y[i] = np.log(X[i] - X[i - 1])

    return Y


def inverse_ordered_vector(Y):
    """Transform from unconstrained space to ordered vector"""

    X = np.empty(Y.size)

    for i in range(Y.size):

        if i == 0:

            X[i] = Y[0]
        
        else:
            X[i] = X[i-1] + np.exp(Y[i])

    return X

def analytic_prior(mu, sigma):
    """Return prior log probability"""

    mu_prior = sts.norm.logpdf(mu, 0, 6) #normal prior on mu
    sigma_prior = sts.lognorm.logpdf(sigma, s = 0.5, scale = np.exp(0)) #lognormal prior of sigma

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
    sigma_prior = sts.lognorm.logpdf(sigma, s = 0.5, scale = np.exp(0)) #lognormal prior of sigma
   
    z_prior = np.sum(sts.norm.logpdf(z, np.array([0, mu]), sigma))

    return mu_prior + sigma_prior + z_prior



def check_ordered(x):

    return x[0] < x[1]



def prior_rvs():

    mu_prior = sts.norm.rvs(0, 6) #normal prior on mu
    sigma_prior = sts.lognorm.rvs(s = 0.5, scale = np.exp(0)) #lognormal prior of sigma
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

    out[1] = out[1]

    return out

def transformed_proposal_distribution(y, theta, cov = 1.0):

    #1) Make into eqivilant z_hat
    #2) Make jump in unconstrained space
    #3) convert back to z

    z_hat = theta[2:][np.argsort(y)]
    z_hat_unconstrained = ordered_vector(z_hat)

    theta_unconstrained = theta.copy()
    theta_unconstrained[2:] = z_hat_unconstrained
      
    out = sts.multivariate_normal.rvs(theta_unconstrained, cov = cov)
    out[1] = np.abs(out[1])

    #Map z_hat back to ordered space
    z_hat_new = inverse_ordered_vector(out[2:])

    #Map z_hat to z    
    z_new = z_hat_new[(y-1)]    
    out[2:] = z_new
    
    pdb.set_trace()
    match = check_rank_match(y, out[2:])

    if not match:
        pdb.set_trace()

    return out

def transformed_prior_rvs(y):

    mu_prior = sts.norm.rvs(0, 6) #normal prior on mu
    sigma_prior = sts.lognorm.rvs(0.5, loc = 0) #lognormal prior of sigma

    #Transformed z_prior

    MU = np.array([0, mu_prior])
    z_prior = sts.norm.rvs(MU[np.argsort(y)], sigma_prior)

    return np.concatenate((np.array([mu_prior, sigma_prior]), z_prior))

    
def censor_prior_transformed(y, mu, sigma, z_hat):


    #Shift to unconstrained space   
    z_hat_new = inverse_ordered_vector(z_hat)    
    z_new = z_hat_new[(y-1)] #The transformed z_hat   


    #Calculate like log likelihood
    mu_prior = sts.norm.logpdf(mu, 0, 6) #normal prior on mu
    sigma_prior = sts.lognorm.logpdf(sigma, s= 0.5, scale = np.exp(0)) #lognormal prior of sigma
    
    #Transformed z_prior
    MU = np.array([0, mu])

    z_prior = sts.norm.logpdf(z_new, MU[np.argsort(y)], sigma).sum() + z_hat[1]    
          
    return mu_prior + sigma_prior + z_prior

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

def check_rank_match(y, z):

    return np.all(rk.rank(z) == y)

def mcmc_censor(N_samples, y):

    #Initialise 
    match = False

    while not match:
        theta_censor = prior_rvs()
        match = check_rank_match(y, theta_censor[2:])

    all_theta = []    

    t = 0
    i = 0
    while i < N_samples:

        t += 1
        if (i %50) == 0:
            print("sample: {}".format(i))

        prop_theta_censor = analytic_proposal_distribution(theta_censor, 0.7)
        match = check_rank_match(y, prop_theta_censor[2:])

        if (not match) or (prop_theta_censor[1] < 0):
            continue

        else:

            p_old = censor_prior(theta_censor[0], theta_censor[1], theta_censor[2:])
            p_new = censor_prior(prop_theta_censor[0], prop_theta_censor[1], prop_theta_censor[2:])

            accept_p = np.min([1, np.exp(p_new - p_old)])

            if np.random.rand() < accept_p:

                theta_censor = prop_theta_censor
                all_theta.append(theta_censor)
                i += 1
   
    return np.array(all_theta), i/t






def mcmc_censor_transformed(N_samples, y):

  
    ordered = False

    #Start at a position in constrained space
    while not ordered:
        theta = transformed_prior_rvs(y)
        ordered = check_ordered(theta[2:])

    #Map to unconstrained space
    theta[2:] = ordered_vector(theta[2:]) 
    theta_censor = theta.copy()
   
     
    all_theta = []    

    t = 0
    i = 0    
    while i < N_samples:  

        t += 1 

        if (i %50) == 0:
            print("sample: {}".format(i))

        prop_theta_censor = analytic_proposal_distribution(theta_censor, [3., 1., 1.5, 1.5])

        p_old = censor_prior_transformed(y, theta_censor[0], theta_censor[1], theta_censor[2:])
        p_new = censor_prior_transformed(y, prop_theta_censor[0], prop_theta_censor[1], prop_theta_censor[2:])
        

        accept_p = np.min([1, np.exp(p_new - p_old)])

        if np.random.rand() < accept_p:

            theta_censor = prop_theta_censor
            all_theta.append(theta_censor)
            i += 1

    return np.array(all_theta), i/t 

if __name__ == '__main__':

    #Generate
    # mu = np.array([0, 0.5])
    # sigma = 1.0
    # y = generate_data(mu, sigma)

    y = np.array([1, 2])
    print(y)
 
    init_theta = np.array([0.5, 1.0])   
    #Metropolis steps
    N_samples = 6000
    warmup = 3000

    
    fit1, acc_p_1 = mcmc_analytic(N_samples, warmup, init_theta, y)    
    fit2, acc_p_2  = mcmc_censor(N_samples, y)
    # fit3 = mcmc_censor_transformed(N_samples, y)

    print(acc_p_2)

    plt.figure()
    plt.plot(fit1[warmup:,0], 'r', alpha = 0.5)
    plt.plot(fit2[warmup:,0], 'b', alpha = 0.5)
    # plt.plot(fit3[warmup:,0], 'g', alpha = 0.5)

    

    plt.figure()
    plt.hist(fit1[warmup:,0], density=True, alpha = 0.5, color = 'r', bins = 20, range = (-5, 20))
    plt.hist(fit2[warmup:,0], density=True, alpha = 0.5, color = 'b', bins = 20, range = (-5, 20))
    
    plt.show()
    
    plt.hist(fit3[warmup:,0], density=True, alpha = 0.5, color = 'g', bins = 20)
    
    plt.figure()
    plt.hist(fit1[warmup:,1], density=True, alpha = 0.5, color = 'r', bins = 20, range = (0, 9))
    plt.hist(fit2[warmup:,1], density=True, alpha = 0.5, color = 'b', bins = 20, range = (0, 9))
    plt.hist(fit3[warmup:,1], density=True, alpha = 0.5, color = 'g', bins = 20, range = (0, 9))


    

    plt.figure()
    plt.plot(fit1[warmup:,1], 'r', alpha = 0.5)
    plt.plot(fit2[warmup:,1], 'b', alpha = 0.5)
    plt.plot(fit3[warmup:,1], 'g', alpha = 0.5)


    plt.show()



