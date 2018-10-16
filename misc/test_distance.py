import sys
rootpath = 'M:\Transport_Studies\Work_Projects\PSI\Publication_Projects\PyThurstonian'
sys.path.append(rootpath)

from PyThurstonian import thurstonian, simulate_data, run_sample, hdi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from multiprocessing import Process #If we want to use multiprocessing

from PyThurstonian import ranking as rk 

data = pd.read_csv("sim_rank_data.csv")


c2 = data[data["Condition"] == 'C1']

Y = c2[['Y1', 'Y2', 'Y3']].values

S = np.array([1,2,3])


d = rk.kendall_tau_dist_vec(Y, np.tile(S, (Y.shape[0],1)), norm = True)

print(d.sum())