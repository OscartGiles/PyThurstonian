import numpy as np 
import scipy.stats as sts 
from PyThurstonian import ranking as rk

import matplotlib.pyplot as plt 
import seaborn as sns

def likelihood(mu1, mu2, sigma):

    return sts.norm.cdf(0, mu1 - mu2, np.sqrt(2 * sigma**2)) 

def assert_params_identical(mu, sigma, N = 10000):

    """There are two ways to parameterise the Thurstonian model. 

    This simulation asserts that the following two models give the same distribution over y:

        Model 1:
            z ~ N(mu, sigma)
            y = rank(z)

        Model 2:
            z ~ N(mu * 1 /sigma, 1)
            y = rank(z)
    """
    tau = 1 / sigma

    K = mu.size    

    #Random variates

    e = sts.norm.rvs(0, 1, size = (N, K))

    z1 = mu + (e * sigma)

    z2 = (mu * tau) + e

    y1 = rk.rank(z1)

    y2 = rk.rank(z2)

    print("All match = {}".format(np.all(y1 == y2)))

    return y1, y2, z1, z2
    

if __name__ == '__main__':

    mu = np.array([3.2, -1., 1.5])
    sigma = 2.0

    y1, y2, z1, z2 = assert_params_identical(mu, sigma)

    
    #Plot the resulting distributions over z
    fig, ax = plt.subplots(1, 2,sharex=True, sharey=True)

    for i in range(z1.shape[1]):
        sns.kdeplot(z1[:,i], ax = ax[0], shade = False)
  
    for i in range(z1.shape[1]):
        sns.kdeplot(z2[:,i], ax = ax[1], shade = False)

    plt.show()

