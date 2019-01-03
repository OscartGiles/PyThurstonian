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

def transformed_prior_rvs(y):

    mu_prior = sts.norm.rvs(0, 6) #normal prior on mu
    sigma_prior = sts.lognorm.rvs(0.5, loc = 0) #lognormal prior of sigma

    #Transformed z_prior

    MU = np.array([0, mu_prior])
    z_prior = sts.norm.rvs(MU[np.argsort(y)], sigma_prior)

    return np.concatenate((np.array([mu_prior, sigma_prior]), z_prior))

def check_ordered(x):

    return x[0] < x[1]

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
    
    
    if not np.isfinite(mu_prior + sigma_prior + z_prior):

        pdb.set_trace()
     

    return mu_prior + sigma_prior + z_prior

def analytic_proposal_distribution(theta, cov = 1.0):

    out = sts.multivariate_normal.rvs(theta, cov = cov)

    out[1] = np.abs(out[1])

    return out


if __name__ == '__main__':


    ordered = False
    
    y = np.array([1,2])

    #Start at a position in constrained space
    while not ordered:
        theta = transformed_prior_rvs(y)
        ordered = check_ordered(theta[2:])


    #Map to unconstrained space
    theta[2:] = ordered_vector(theta[2:])
 
    theta_censor = theta.copy()
   
   
    all_theta = []    

    N_samples = 10000
    i = 0    
    while i < N_samples:  

        if (i % 20) == 0:
            print("sample: {}".format(i))

        prop_theta_censor = analytic_proposal_distribution(theta_censor)


        p_old = censor_prior_transformed(y, theta_censor[0], theta_censor[1], theta_censor[2:])
        p_new = censor_prior_transformed(y, prop_theta_censor[0], prop_theta_censor[1], prop_theta_censor[2:])
        
        if (not np.isfinite(p_old)) or (not np.isfinite(p_new)):
                pdb.set_trace()
        #Map back to standard model

        # theta_standard = theta_censor.copy()
        # theta_standard[2:] = inverse_ordered_vector(theta_standard[2:])

        # prop_theta_standard = prop_theta_censor.copy()
        # prop_theta_standard[2:] = inverse_ordered_vector(prop_theta_standard[2:])
     
        accept_p = np.min([1, np.exp(p_new - p_old)])

        if np.random.rand() < accept_p:

            theta_censor = prop_theta_censor
            all_theta.append(theta_censor)
            i += 1
   
    pdb.set_trace()
    # return np.array(all_theta)
