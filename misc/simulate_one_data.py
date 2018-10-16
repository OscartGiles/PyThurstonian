#This is just for working with PyThurstonian on my machine. Should remove on release
import sys
rootpath = 'M:\Transport_Studies\Work_Projects\PSI\Publication_Projects\PyThurstonian'
sys.path.append(rootpath)


from PyThurstonian import simulate_data, thurstonian
from PyThurstonian import ranking as rk 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

import pdb


K = 2 # Number of items being ranked
J = 1 # Number of subjects
L = 1000 #Number of trials (Subjects can contribute more than one trial to the same condition)
C = 1 #Number of conditions (Number of conditions. Here we assume a within subjects design, but PyThurstonian will automatically handle different design types)

#Beta parameters
beta = np.array([[0.0, 0.4]]) 
data, sim_scale = simulate_data(K, J, L, C, beta = beta, scale = np.array([1.])) #Set a seed to recreate the same dataset

Y = data[['Y1', 'Y2']].values

data['Subj'] = pd.factorize(data['Subj'])[0]+1
myThurst = thurstonian(design_formula = '~0+Condition', data = data, subject_name = "Subj")   
    
# print(data)
data.to_csv("misc/rank_data.csv")


print(rk.kendall_W(Y))
