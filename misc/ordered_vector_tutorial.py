import numpy as np 
import scipy.stats as sts 
import matplotlib.pyplot as plt 
import sys
import pdb 

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

  
def sample_z(mu, sigma, k, seed = None):

    if seed is not None:
        np.random.seed(seed)
    mu = np.array([-1, 1, 0])
    sigma = 1.0

    z = sts.norm.rvs(mu, sigma)
    y = rk.rank(z)

    return z, y

def sample_z_hat(mu, sigma, k, seed = None):

    if seed is not None:
        np.random.seed(seed)
    mu = np.array([-1, 1, 0])
    sigma = 1.0


    z = sts.norm.rvs(mu, sigma)
    y = rk.rank(z)

    x = np.argsort(y)
    x_hat = np.argsort(x)

    z_hat = z[x]
    z_new = z_hat[x_hat]
   
    print(np.all(z == z_new))
    assert np.all(z == z_new)

    return z_new, y




if __name__ == '__main__':


  
    # mu = np.array([-1, 1, 0])

    # sigma = 1.0

    # # z, y = sample_z(mu, sigma, 3, seed = 5434745)

    # for i in range(1000):
    #     z, y = sample_z_hat(mu, sigma, 3)
    # print(z)
    # print(y)

    ov = np.array([-2, 4, 10.2, 15.3])

    ov_t = ordered_vector(ov)

    ov_new = inverse_ordered_vector(ov_t)

    print(ov)
    print(ov_t)
    print(ov_new)