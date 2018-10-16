# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 10:39:11 2017

@author: Oscar Terence Giles
@Email: o.t.giles@leeds.ac.uk


Convienience functions for ranking data. 
"""

import numpy as np
from math import factorial
from itertools import permutations, combinations

import pdb

def rank(x, axis = -1):
    """Rank the array x along a specified axis
    
    args:
        x: N-D array of floats of ints
        axis: The axis to rank over
        
    Details:
        Does the same as scipy.stats.rank but allows for operation over different dimensions of array. scipy only ranks a 1D array"""

    temp = np.argsort(np.argsort(x, axis = axis), axis = axis)
    return temp + 1

def borda_count(x, axis = -1):
    """Borda count of the N-D array x, along the specified axis.
    
    args: 
        x: N-D array of rank data (integers)
        axis: The axis to perform the operation over
    
    Details:
        First sums over an axis and then ranks by the sums"""    
    
    return rank(np.sum(x, axis = axis), axis = axis)



def assign_numeric_rank(x, ignore_warn = False):
    """Assign a number to each possible unique ranking and then categorise x by it.
    
    For example, when categorising 3 items there are 3! possible rankings. The function categorises them as 1 to 6 and returns a key
    
    args:
        x: An N-D array where the last dimension is the ranks
        ignore_warn: When the number of possible rankings is greater than 20 the function will raise an error. Turn ignore_warn to False to avoid this. It may take a very long time for this function to return in these cases. Not advised
    
    """    
    
    K = x.shape[-1]
    if not ignore_warn:
        if factorial(K) > 24:
            raise RuntimeError("There are a lot of possible permutations. If you really want to run set 'ignore_warn = True'" )
                  
    poss_perms = np.array(list(permutations(range(1, K+1)))) 
      
    return np.array([np.argmax(np.all(poss_perms == i, axis = 1)) + 1 for i in x]), poss_perms


def kendall_tau_dist(A, B, norm = True):
    """Return the kendall tau distance between A and B.
            
    args:
        A: A 1-D array of rankings
        B: A 1-D array of rankings
        norm: Normalse the tau distance so it is between 0 and 1"""
    
    if not (isinstance(A, list) and isinstance(B, list)):
        
        try:
            A = list(A)            
            B = list(B)            
        except:            
            return np.NaN
    
    pairs = combinations(range(0, len(A)), 2)
    distance = 0

    for x, y in pairs:
        a = A[x] - A[y]
        b = B[x] - B[y]

        # if discordant (different signs)
        if (a * b < 0):
            
            distance += 1

    if norm:
        
        distance = distance / max_tau_dist(len(A))

    return distance

def kendall_tau_dist_vec(A, B, norm = True):
    """Return the kendall tau distance between A and B, where A and B and NxK vectors
    
        
    args:
        A: A NxK array of rankings
        B: A NxK array of rankings
        norm: Normalse the tau distance so it is between 0 and 1"""
    
   
    
    pairs = combinations(range(0, A.shape[1]), 2)
    distance = np.zeros(A.shape[0])
  
    
    for x, y in pairs:
        a = A[:,x] - A[:,y]
        b = B[:,x] - B[:,y]
        
#        pdb.set_trace()
        # if discordant (different signs)        
        distance[(a * b) < 0 ] = distance[(a * b) < 0 ] + 1

    if norm:
        
        distance = distance / max_tau_dist(A.shape[1])

    return distance

def kendall_W(Y):
    """Calculate kendall's W, where Y is an J X K matrix of J judges and K items"""

    
    J = Y.shape[0]
    K = Y.shape[1]

    R_k = np.sum(Y, axis = 0)    
    R_mean = np.mean(R_k)

    S = np.sum(np.square(R_k - R_mean))

    W = (12 * S) / (J**2 * (K**3 - K))

    return W

def max_tau_dist(K):
    """Return the maximum unnormalised distance of K rankings. In other words, the maximum number of swaps performed by the bubble sort algorithm to reverse a list of rankings"""
    
    return (K * (K - 1)/2 )
    

def bin_taus(tau, K, norm = True):
    """Given a list of taus return a bincount of taus for plotting as a bar plot (for use in cases when the possible number of taus is small)"""
      
    tau = np.array(tau)
        
    bins = np.array(list(range(int(max_tau_dist(K)) + 1 ))) 
    
    if norm:
        bins = bins / max_tau_dist(K)
    
    counts = np.zeros(bins.shape[0])
    
    for i, b in enumerate(bins):        

        counts[i] = np.isclose(tau, b).sum()

    str_bins = np.round(bins, decimals = 3).astype(str)
    return str_bins, counts

def bin_ranks(ranks, K, ignore_warn = False):
    """Return a bin count of the number of ranks"""
    
    if not ignore_warn:
        if factorial(K) > 24:
            raise RuntimeError("There are a lot of possible permutations. If you really want to run set 'ignore_warn = True'" )
    
    poss_perms = list(permutations(range(1, K+1)))
    str_perms = ["-".join(map(str, i)) for i in poss_perms]
    
    counts = np.zeros(len(poss_perms))
    
    for i in range(len(poss_perms)):
        counts[i] = np.sum(np.all(ranks == poss_perms[i], axis = -1))
    
    
    return str_perms, counts


if __name__ ==  '__main__':
    
    r = np.array([[1,2,3], 
              [1,2,3],
              [2,1,3], 
              [3,1,2], 
              [3,1,2], 
              [3,2,1]])
        
    out = bin_ranks(r, 3)
    #bin_taus([0.33333333, 0.6666666, 1, 1, 0, 0, 0], 3)
