# -*- coding: utf-8 -*-
"""
Created on Mon May 14 11:03:18 2018

@author: pscog
"""

from PyThurstonian import thurstonian, run_sample
from simulate_data import simulate_data
import ranking as rk
import numpy as np
import pandas as pd

from multiprocessing import Process


if __name__ == '__main__':
    
    #Load and preprocess data  
    K = 3 #Number of items
    J = 10 #Number of participants
    L = 1 #Number of trials
    C = 2 #Number of conditions
    
    beta = np.array([[0.0, 1.0, 2.0],
                     [0.0, 0.1, 0.2]])        
    
    data, sim_scale = simulate_data(K, J, L, C, beta = beta, seed = 4354356)
    
    data['Subj'] = pd.factorize(data['Subj'])[0]+1
    myThurst = thurstonian(design_formula = '~0+Condition', data = data, subject_name = "Subj")   
    
    myThurst.pre_sample()
    
    
    P = [Process(target=run_sample, kwargs = {'temp_fnum': i})  for i in range(4)] 
            
    for p in P:
        p.start()
        
    for p in P:
        p.join()
        
    myThurst.post_sample()