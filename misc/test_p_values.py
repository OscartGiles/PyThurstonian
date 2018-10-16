# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 13:04:57 2018

@author: pscog
"""



#import sys
#import os

#sys.path.append(os.path.dirname(os.getcwd()))

from multiprocessing import Process, Pool

import PyThurstonian
from PyThurstonian import thurstonian, hdi

import pandas as pd
import seaborn as sns   
import matplotlib.pyplot as plt
import numpy as np
from simulate_data import simulate_data
import ranking as rk

import pdb
    



#Set plotting preferences
sns.set_context("paper")    
sns.set_style("whitegrid")

output_folder = "article/Figures/"

#Load and preprocess data  
K = 3 #Number of items
J = 30 #Number of participants
L = 1 #Number of trials
C = 2 #Number of conditions

beta = np.array([[0.0, 1.0, 2.0],
                 [0.0, 0.0, 0.0]])        

N_samples = 5000


def get_p(i):
    
    data, sim_scale = simulate_data(K, J, L, C, beta = beta)

    data['Subj'] = pd.factorize(data['Subj'])[0]+1
    myThurst = thurstonian(design_formula = '~0+Condition', data = data, subject_name = "Subj")   
        
    return myThurst.sra_p_value(level_dict = {'Condition':  'C2'})[0]

if __name__ == '__main__':
    
    with Pool(6) as p:
        all_p = p.map(get_p, range(5000))

    all_p = np.array(all_p)
    
    print(all_p)
    
    print(np.sum(all_p < 0.05) / all_p.size)
    
    plt.hist(all_p, normed = True)
    plt.show()