# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 16:33:00 2017

@author: pscog
"""

import pandas as pd
import numpy as np
import csv
import pdb

def join_files(files, convert_to_float = True):
    """Join all the files created by cmdstan and join them. Return as a Pandas dataframe
    
    args:
        convert_to_float: Return all the dataframe columns as floats. Otherwise they will be strings (annoying)"""
    
    master_data = pd.DataFrame()
    
    for fi in files:        
   
        with open(fi) as f:
            
         
            data = [r for r in csv.reader(f, delimiter=',') if r[0][0] != '#']            
            data = pd.DataFrame.from_records(data[1:], columns = data[0])
            
#            data['chain_id'] = i          
           
            master_data = pd.concat((master_data, data), ignore_index = True)        

    if convert_to_float:
        return master_data.astype(float)
    else:
        return master_data

def check_divergent(data):
    
    n_divergent = data['divergent__'].sum() 
    
    if n_divergent > 0:
        
        print("Warning: {} divergent step(s) detected. The sampler may be biased. Try increasing adapt_delta toward 1.".format(int(n_divergent)))
     
        return True
    
    else:
        return False
        
def get_params(data):
    """Pass a pandas data frame of samples and return a dictionary of parameters"""
    
    
#    del data['chain_id']    
    cols = data.columns            
    
    parameters = list(set([i.split(".")[0] for i in data.columns[7:]])) #Get all the unique parameters    
    
    #Now get the number of dimensions for each parameter
    param_dict = {}
    
    for par in parameters:
        
        
        all_par_cols = [i for i in cols if par + '.' in i] #Get all the column names for the parameter par. Add the + '.' to make sure file is unique. This is a fudge. may fail in some cases       
     
        par_dims = np.array([i.split(".")[1:] for i in all_par_cols], dtype = int).max(axis = 0) #Get the dimensions of the parameter
    
        #Now reshape the array        
        param_vals = data[all_par_cols].values.reshape(tuple(np.append(data.shape[0], par_dims)), order = "F")
        
        param_dict[par] = param_vals
    
    return param_dict
        
        
    
