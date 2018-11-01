# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 13:04:57 2018

@author: Oscar T Giles
@email: o.t.giles@leeds.ac.uk
"""

from . import ranking as rk
from . import misc
from . import process_samples

import os, pdb, pickle, glob, itertools
import subprocess as sp

import pandas as pd
import numpy as np
import patsy
import matplotlib.pyplot as plt
import seaborn as sns 

from tkinter import filedialog, Tk

# import ranking

def make_indices(dimensions):
    # Generates complete set of indices for given dimensions
    level = len(dimensions)
    if level == 1:
        return list(range(dimensions[0]))
    indices = [[]]
    while level:
        _indices = []
        for j in range(dimensions[level - 1]):
            _indices += [[j] + i for i in indices]
        indices = _indices
        level -= 1
    try:
        return [tuple(i) for i in indices]
    except TypeError:
        return indices
    
def calc_min_interval(x, alpha):
    """Internal method to determine the minimum interval of a given width
    Assumes that x is sorted numpy array.
    """

    n = len(x)
    cred_mass = 1.0-alpha

    interval_idx_inc = int(np.floor(cred_mass*n))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError('Too few elements for interval calculation')

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx+interval_idx_inc]
    return hdi_min, hdi_max


def hdi(x, alpha=0.05):
    """Calculate highest posterior density (HPD) of array for given alpha. 
    The HPD is the minimum width Bayesian credible interval (BCI).
    :Arguments:
        x : Numpy array
        An array containing MCMC samples
        alpha : float
        Desired probability of type I error (defaults to 0.05)
    """

    # Make a copy of trace
    x = x.copy()
    # For multivariate node
    if x.ndim > 1:
        # Transpose first, then sort
        tx = np.transpose(x, list(range(x.ndim))[1:]+[0])
        dims = np.shape(tx)
        # Container list for intervals
        intervals = np.resize(0.0, dims[:-1]+(2,))

        for index in make_indices(dims[:-1]):
            try:
                index = tuple(index)
            except TypeError:
                pass

            # Sort trace
            sx = np.sort(tx[index])
            # Append to list
            intervals[index] = calc_min_interval(sx, alpha)
        # Transpose back before returning
        return np.array(intervals)
    else:
        # Sort univariate node
        sx = np.sort(x)
        return np.array(calc_min_interval(sx, alpha))
    
    
class thurstonian:
    """A class for fitting Thurstonian rank models using PyStan"""
    
    def __init__(self, design_formula, data, subject_name = 'Subj'):
        

        self.__module_dir = os.path.dirname(__file__) #Get the module directory
        self.__stan_model_dir = os.path.join(self.__module_dir, 'stan_model')
        self.__stan_model_path = os.path.join(self.__module_dir, 'stan_model', 'thurstonian_cov.exe') #Location of the stan model


        self.design_formula = design_formula
        self.data = data        
        self.subj_name = subject_name            
    
        self.X = self.__gen_design_matrix()        
        self.y = self.__gen_y_matrix()
        self.subj = self.__gen_subj_vec()       
        
        self.N = self.y.shape[0]
        self.K = self.y.shape[1]
        self.C = self.X.shape[1]
        self.J = np.unique(self.subj).shape[0]        
        
        self.__check_stan_model_compiled() #Ensure the stan model is compilled

        
        
    def __gen_design_matrix(self):
        """Return a design matrix"""
        
        return patsy.dmatrix(self.design_formula, data = self.data)
    
    def __gen_y_matrix(self):
        """Return the data in a matrix"""
        
        return self.data[[i for i in self.data.columns if 'Y' in i]].values
    
    def __gen_subj_vec(self):
        
              
        return self.data[self.subj_name]
    
    def __gen_stan_data(self, priors, rep_data = 1):
        """Return a dictionary of data for stan"""
                
        return {'N': self.N, 'K': self.K, 'C': self.C, 'J': self.J, 'X': self.X, 'y': self.y, 'rater': self.subj.values, 'beta_sd_prior': priors['beta_sd_prior'], 
                 'scale_sd_prior': priors['scale_sd_prior'], 'rep_data': rep_data}
        
    def __check_stan_model_compiled(self):
        """Check the stan model is compiled, else compile it"""
        
        if os.path.exists(self.__stan_model_path):   
            
            pass
                 
        else:         
            print("PyThurstonian requires a Stan model to be compilled the first time it is run")               
            self.__compile_stan_model()
        
        
    def __compile_stan_model(self):
        """Compile the stan model. Asks the user to set the location of cmdstan        
        
        This requires a c++ compiler and cmdstan (http://mc-stan.org/users/interfaces/cmdstan) to be installed.
        Follow the cmdstan installtion instructions"""
        
               
        root = Tk()
        root.withdraw()
        root.directory =  filedialog.askdirectory(title = "Set the location of cmdstan")
                       
        self.__build_stan_model(root.directory)
        

    def __build_stan_model(self, directory):
        """Attempt to build cmdstan
        
        This will fail if script isnt in the Pythurstonian directory. Needs a fix"""

                       
        model_path = self.__stan_model_path
        model_path = model_path.replace('\\', '/')
    
        print("Compilling. This may take a while")
        # pdb.set_trace()
        a = sp.check_output(['make', model_path], 
                        cwd = directory)
            
    def __rep_data_boolean(self, x):
        
        if x:
            return 1
        else:
            return 0
        
    def pre_sample(self, priors = None, rep_data = True):
        """Prepare for sampling.
        
        This will delete any previous temporary sampling files"""

        
        if priors == None:
            
            priors = {'beta_sd_prior': self.K*3, 'scale_sd_prior': 0.5}                
        
        stan_data = self.__gen_stan_data(priors, rep_data = self.__rep_data_boolean(rep_data))       
        
        #Write the standata to file
        misc.stan_rdump(stan_data, os.path.join(self.__stan_model_dir, "temp.data.R")) #Dump the stan data to a temp file       
        
        #Delete any previous samples        
        all_temp_samples = glob.glob('{}/temp_sample_file*'.format(self.__stan_model_dir))
        
        print('Found {} existing temporary sample files. Removing files'.format(len(all_temp_samples)))
        
        for f in all_temp_samples:            
            os.remove(f)
     
                
    
    
    def post_sample(self, write_sample_file = None):
        
       #Get the data back into Python
        chain_files = glob.glob("{}/temp_sample_file*".format(self.__stan_model_dir))        
        self._diagnostic_samples = process_samples.join_files(chain_files) #Convert everything to a float    
        
      
        if write_sample_file is not None:
            
            self._diagnostic_samples.to_csv(write_sample_file, index = False)
            
        self.divergent_found = process_samples.check_divergent(self._diagnostic_samples)
        
        self.samples = process_samples.get_params(self._diagnostic_samples)
        self.__extract_samples()
             
        
    def sample(self, iter = 2000, warmup = None, adapt_delta = 0.99, priors = None, rep_data = True, refresh = 250, write_sample_file = None):
        """Sample from the Thurstonian model using Stan.
        
        Arguments:
            
            iter: Integer. The number of iterations per chain (pystan arg)
            warmup: Integer. The number of warmup iterations (pystan arg)
            chains: Integer. The number of chains to run (default = 1; pystan arg)
            n_jobs: Integer. The number of cores to use. Must be set to 1 on Windows (default = 1; pystan arg)
            adapt_delta: Delta adaption parameter for NUTS
            priors: Dictionary of model priors
            rep_data: Boolean. 
        """                
        
        #Prepare the data for sampling
        self.pre_sample(priors, rep_data)
        
        #Run the sampling
        run_sample(iter = iter, warmup = warmup, adapt_delta= adapt_delta, refresh = refresh) 

          
        self.post_sample(write_sample_file = write_sample_file)
    
    
    def summary(self):
        """print a summary of the MCMC fit"""
        
        if not hasattr(self, 'fit'):              
            print("No fit instance found")        
        else:
            print(self.fit)        
    
    def __extract_samples(self):
        
        self.beta = self.samples['beta']   
        self.scale = self.samples['scale']        
        
    def save_samples(self, file_name):
        """Save the samples as a pickle object so they can be reloaded and used later.
        
        file_name: str. A file name for your mcmc samples (e.g. 'mysamples')"""       
        
        #Check we have some samples to save        
        if hasattr(self, 'samples'):        
            with open("{}.pkl".format(file_name), 'wb') as f:
                pickle.dump(self.samples, f)                
        else:
            raise AttributeError("No MCMC samples exist. Please call sample() to sample from the posterior.")        
                
    def load_samples(self, file_name):
        """Reload pickled samples so they can be reused"""
        
        with open("{}.pkl".format(file_name), 'rb') as f:
            self.samples = pickle.load(f)   
            self.__extract_samples()
    
        
    def get_levels(self, level_dict):
        
        
        level_dict_ = {}
        #Make sure all dictionary items are lists
        for key, value in level_dict.items():
            
            level_dict_[key] = [value]
            
       
        return np.asarray(patsy.build_design_matrices([self.X.design_info], data = pd.DataFrame(level_dict_))).squeeze()
    
    def __multiply_X_beta(self, X):
        """
        X: a design matrix
        beta: a NXJXK array where N is the number of samples, J is the number of terms and K is the number of items"""

        #This is a fudge which could theoretically lead to unexpected behaviour. Needed because einsum fails when there is a squeezable dimension in X
        if self.X.shape[1] == 1:
            return self.beta
        
        #This trick is equivilant to calculating np.dot(X, self.beta[i]) over all i samples. Its just much faster.
        return np.einsum('j,njk->nk', X, self.beta)    
    
    
    
    #####################################
    #-----------------------------------#
    #----------Data functions-----------#
    #-----------------------------------#
    #-----------------------------------#
    #####################################
    
    def pairwise_tau_distance(self, level_dict_1, level_dict_2):
        """Compute the pairwise kendal tau distance between two conditions secified by level_dict_1 and level_dict_2.
        This will only work when subjects provided a single datapoint in each of the two conditions. If this not the case returns NaN.
                        
        Note: The checking that participants are in both conditions is flawed. Make sure you know this to be true"""
        
        
        # 1) Check that subjects provided a datapoint in each of the conditons
        #Prep results/ perform contrast            
        X_cond_1 = self.get_levels(level_dict_1) #Get the condition of the levels 
        X_cond_2 = self.get_levels(level_dict_2) #Get the condition of the levels
        
               
        mask_1 = np.all(self.X == X_cond_1, axis = 1)  
        mask_2 = np.all(self.X == X_cond_2, axis = 1)  
        
        subj_in_1 = self.subj[mask_1].values
        subj_in_2 = self.subj[mask_2].values
        
        subj_in_1.sort()
        subj_in_2.sort()
        
        if np.all(subj_in_1 == subj_in_2):
            
            
            subj_unique = self.subj.unique()
            all_tau = np.empty(subj_unique.size)
            
            for i, id_ in enumerate(subj_unique):
                
                
                m1 = np.all(self.X == X_cond_1, axis = 1) & (self.subj.values == id_)
                m2 = np.all(self.X == X_cond_2, axis = 1) & (self.subj.values == id_)
            
                y_1 = self.y[m1]
                y_2 = self.y[m2]
            
                all_tau[i] = rk.kendall_tau_dist(y_1.flatten(), y_2.flatten())
              
            return all_tau
            
        else:
            
            print("Could not perform pairwise analysis. Subjects did not contribute a single data point to each condition")
            return np.NaN
             
    def average_pairwise_tau_distance(self, level_dict_1, level_dict_2):
        
        tau_dist = self.pairwise_tau_distance(level_dict_1, level_dict_2)

        return np.mean(tau_dist)
    
    
    def __sample_variance(self, y):
        
        av_y = y.mean(0)
        N = y.shape[0]
        
        return np.sum(np.square(y - av_y), axis = 0) / (N - 1)
    
    def __sra(self, y):
        
        variances = self.__sample_variance(y)       
        
        return np.sum(variances) / variances.size       
    
    
    def sample_variance(self, level_dict):
        """Get the sample variance across each item for a given condition"""
        
        X_cond = self.get_levels(level_dict) #Get the condition of the levels        
        mask = np.all(self.X == X_cond, axis = 1)        
        y_cond = self.y[mask] #Get the data for that condition
        
               
        return self.__sample_variance(y_cond)
    
    def sra(self, level_dict):
        """SRA measure adapted from from Erkstom, Gerds and Jenson (2018). Provides a measure of the agreement in a set of rankings.
        Will be 0 when complete agreement"""
        
        X_cond = self.get_levels(level_dict) #Get the condition of the levels        
        mask = np.all(self.X == X_cond, axis = 1)        
        y_cond = self.y[mask] #Get the data for that condition
                
        return self.__sra(y_cond)
                    
   
    def random_rankings(self, K,  J):
        """Generate dataset by randomly permuting rankings"""        
        
        return np.array([np.random.permutation(range(K)) + 1 for i in range(J)])
                        
    
    def sra_p_value(self, level_dict, N_samples = 1000):
        """level_dict: The level of the condition
            N_samples: The number of data simulations used to calculate the p_value            
        """
        
        X_cond = self.get_levels(level_dict) #Get the condition of the levels        
        mask = np.all(self.X == X_cond, axis = 1)        
        y_cond = self.y[mask] #Get the data for that condition   
              
        sample_sra = self.__sra(y_cond)       
        sra_random_dist = [self.__sra(self.random_rankings(y_cond.shape[1], y_cond.shape[0])) for i in range(N_samples)]
        
        p_value = (np.abs(sample_sra) >= np.abs(sra_random_dist)).sum() / np.array(sra_random_dist).size
        
        
        return p_value, sample_sra, sra_random_dist
        
    
    def rep_sra(self, level_dict):
        """Calculate SRA for replicated datasets"""
        
        y_rep = self.generate_reps()      
        
        #Get the condition replications
        X_cond = self.get_levels(level_dict) #Get the condition of the levels     
        mask = np.all(self.X == X_cond, axis = 1)          
        y_rep_cond = y_rep[:, mask]
               
        #This should be vectorised for speed up
        sample_sra = np.array([self.__sra(y_rep_cond[i]) for i in range(y_rep_cond.shape[0]) ] )
        
        hdi_ = hdi(sample_sra)
  
        return sample_sra, hdi_
    

    def predictive_prob_agreement(self, level_dict):
        """Trying to figure out a decent measure of agreement
        
        What is the probability of randomly drawing two individuals with the same ranking from a population"""

        y_rep = self.generate_reps()      
        
        #Get the condition replications
        X_cond = self.get_levels(level_dict) #Get the condition of the levels     
        mask = np.all(self.X == X_cond, axis = 1)          
        y_rep_cond = y_rep[:, mask]

        n_matches = 0
        for i in range(y_rep.shape[0]):
            choice_idx = np.random.choice(y_rep_cond.shape[1], size = 2, replace = False)
            a = y_rep_cond[i, choice_idx]
            match = np.all(np.diff(a, axis = 0) == 0)
            
            if match:
                n_matches += 1

        p_match = n_matches / y_rep_cond.shape[0]
        return p_match

    def plot_sra(self, level_dict, ax = None):
        """Plot the posterior SRA against a random SRA.
        Also shows p-value.
        
        Note: Cant seem to make multiple calls to get condition mask. Needs fixing"""
        
        if ax == None:        
            ax = plt.gca()
            
     
        rep_sra, hdi = self.rep_sra(level_dict)
        p_val, sample_sra, random_sra = self.sra_p_value(level_dict)
        
        
        ax.hist(rep_sra, color = 'r', alpha = 0.25, density = True)
        ax.hist(random_sra, color = 'b', alpha = 0.25, density = True)
        
        ax.axvline(sample_sra, color = 'k', linestyle = '--')
        
        ax.text(0.05, 0.9, 'p = {}'.format(p_val), transform = ax.transAxes)
    

    def plot_kendall_W(self, level_dict, ax  = None, text_x_pos = 0.05):
        """Plot the posterior predictive density over kendall's W"""
        if ax == None:        
            ax = plt.gca()

        y_rep = self.generate_reps()      
        
        #Get the condition replications
        X_cond = self.get_levels(level_dict) #Get the condition of the levels     
        mask = np.all(self.X == X_cond, axis = 1)          
        y_rep_cond = y_rep[:, mask]
        y_cond = self.y[mask]

        W_sample = rk.kendall_W(y_cond)

        W = np.empty(y_rep_cond.shape[0])
        for i in range(y_rep_cond.shape[0]):
            W[i] = rk.kendall_W(y_rep_cond[i])

        ax.hist(W, color = 'r', alpha = 0.25, density = True)
        ax.axvline(W.mean(), color = 'k', linestyle = '--')
        ax.axvline(W_sample, color = 'r', linestyle = '--')
       
        ax.text(text_x_pos, 0.9, 'W = {:.2f}'.format(W_sample), transform = ax.transAxes)
        ax.text(text_x_pos, 0.8, 'W ppd = {:.2f}'.format(W.mean()), transform = ax.transAxes)

        ax.set_xlabel("W")
        ax.set_ylabel("Density")
    #####################################
    #-----------------------------------#
    #--------Plotting Functions---------#
    #-----------------------------------#
    #-----------------------------------#
    #####################################
    
    def generate_reps(self, N_reps = 1):        
        """Generate replicated dataset"""
        
        res = np.random.normal(0, 1, size = (self.beta.shape[0],  self.X.shape[0],  self.K-1, N_reps)) 
              
        sigma = 1/self.scale
        subj_zero = self.subj - 1 #Zero indexed subj            
        sigma_i = sigma[:,subj_zero]    
        sigma_i = np.repeat(sigma_i[:,:,np.newaxis], self.K-1, axis = -1) #repeat over items       
        sigma_i = np.repeat(sigma_i[:,:,:,np.newaxis], N_reps, axis = -1) #Repeat over psuedo_reps

        #Generate errors
        e = res * sigma_i 

        mu = np.einsum('rj,njk->nrk', self.X, self.beta) 
        
        mu_tile = np.repeat(mu[:,:,:,np.newaxis], N_reps, axis = -1)           
    
        z_rep = np.zeros(mu_tile.shape)
        z_rep[:,:, 1:,:] = mu_tile[:, :, 1:, :] + e    
        y_rep = rk.rank(z_rep, axis = 2)
                       
        return y_rep.squeeze()

    
    def plot_data(self, level_dict, ax = None):
        """To plot the data for a condition pass a dictionary of factors and the required levels
        
        Developer note: This will only work for categorical predictors"""
        
        if ax == None:        
            ax = plt.gca()            
         
        X_cond = self.get_levels(level_dict) #Get the condition of the levels        
        mask = np.all(self.X == X_cond, axis = 1)        
        y_cond = self.y[mask] #Get the data for that condition
        
        data_bin = rk.bin_ranks(y_cond, self.K)
        
      
        colors = np.repeat('#FFC0CB', len(data_bin[0]))   
        linewidth = [0.2 for i in range(len(data_bin[0]))]                   
        edgecolor = ['k' for i in range(len(data_bin[0]))]
    
        ax.bar(data_bin[0], data_bin[1] / data_bin[1].sum(), color = colors, linewidth = linewidth, edgecolor = edgecolor)
        
        #Set axis details
        plt.sca(ax)
        plt.xticks(rotation=60)
        plt.xlabel("Rank")
        plt.ylabel("Proportion data")
    
        
    
    def plot_aggregate(self, level_dict, ax = None):
        """Plot the aggregate ranking for a condition"""
        
        if ax == None:        
            ax = plt.gca()
            
        X_cond = self.get_levels(level_dict) #Get the condition of the levels  
        
        cond_samples = self.__multiply_X_beta(X_cond)  
        
        cond_rank = rk.rank(cond_samples, axis = -1)
        
        rank_bin = rk.bin_ranks(cond_rank, cond_rank.shape[-1])
        
        max_p = np.argmax(rank_bin[1])
        line_widths = [0.2 for i in range(len(rank_bin[0]))]  
        line_widths[max_p] = 1.0
        colors = np.repeat('#E4F1FE', len(rank_bin[0]))
        colors[max_p] = '#22313F'
        
        ax.axhline(0.05, color = 'k', linestyle = '--', alpha = 0.25)
                
        ax.bar(rank_bin[0], rank_bin[1] / rank_bin[1].sum(), color = colors, 
          edgecolor = ['k' for i in range(len(rank_bin[0]))],
          linewidth = line_widths)  
              
        
        #Set axis details
        plt.sca(ax)
        plt.xticks(rotation=60)
        plt.xlabel("Rank")
        plt.ylabel("P(Rank)")
           
        return rank_bin 

    def get_mean_ranks(self, level_dict):
        """Get the mean ranks of a certain level"""
        
        X_cond = self.get_levels(level_dict) #Get the condition of the levels  
        
        cond_samples = self.__multiply_X_beta(X_cond)  
               
        return rk.rank(cond_samples.mean(0), axis = -1)
    
    def get_p_distance(self, level_dict, alpha):
        """Get the tau distance which contains alpha percent of the ranks"""
        
        X_cond = self.get_levels(level_dict) #Get the condition of the levels  
        
        cond_samples = self.__multiply_X_beta(X_cond)  
        
        cond_rank = rk.rank(cond_samples, axis = -1)
        mean_rank = rk.rank(cond_samples.mean(0), axis = -1)
        
        #Get rid of single dimensions
        cond_rank = cond_rank.squeeze()
        mean_rank = mean_rank.squeeze()
        
        tau_dist = rk.kendall_tau_dist_vec(np.tile(mean_rank, (cond_rank.shape[0],1)), cond_rank)
        
        return tau_dist[tau_dist.argsort()][int(tau_dist.shape[0] * alpha)]
        
        
        
    def plot_aggregate_complex(self, level_dict, ax = None):
        """Doesn't make sense to plot this???"""
        
        if ax == None:        
            ax = plt.gca()
            
        X_cond = self.get_levels(level_dict) #Get the condition of the levels  
        
        cond_samples = self.__multiply_X_beta(X_cond)  
        
        cond_rank = rk.rank(cond_samples, axis = -1)
        mean_rank = rk.rank(cond_samples.mean(0), axis = -1)
        
        #Get rid of single dimensions
        cond_rank = cond_rank.squeeze()
        mean_rank = mean_rank.squeeze()

        
        tau_dist = rk.kendall_tau_dist_vec(np.tile(mean_rank, (cond_rank.shape[0],1)), cond_rank)
            
        max_bin = rk.max_tau_dist(self.K)
        
        if max_bin < 20:
            tau_bins = rk.bin_taus(tau_dist, self.K)  
        
            tau_bins_0 = [float(i) for i in tau_bins[0]]
            width = np.diff(tau_bins_0).min()
            
            #Create the plot
            colors = np.repeat('#E4F1FE', len(tau_bins[0]))   
                            
            ax.bar(tau_bins[0], tau_bins[1] / tau_bins[1].sum(), width,
                          color = colors, edgecolor = ['k' for i in range(len(tau_bins[0]))]) 
            
        else:
            
            ax.hist(tau_dist, alpha = 1, color = '#E4F1FE', density = True)
            sns.kdeplot(tau_dist, ax = ax, color = '#22313F')
                        

        #Set axis details
        plt.sca(ax)        
        plt.xlabel(r"$\tau_{kendall}$")
        plt.ylabel(r"$P(\tau_{kendall})$")
               
    
    def plot_contrast(self, level_dict_1, level_dict_2, ax = None):        
        """Contrast between two conditions"""
        
        if ax == None:        
            ax = plt.gca()            
        
        #Prep results/ perform contrast            
        X_cond_1 = self.get_levels(level_dict_1) #Get the condition of the levels 
        X_cond_2 = self.get_levels(level_dict_2) #Get the condition of the levels
        
        cond_samples_1 = self.__multiply_X_beta(X_cond_1) 
        cond_samples_2 = self.__multiply_X_beta(X_cond_2) 
        
        cond_rank_1 = rk.rank(cond_samples_1, axis = -1)
        cond_rank_2 = rk.rank(cond_samples_2, axis = -1)
        
        comp_taus = rk.kendall_tau_dist_vec(cond_rank_1, cond_rank_2)       
        comp_bins = rk.bin_taus(comp_taus, self.K)    
    
    
        #Create the plot
        colors = np.repeat('#E4F1FE', len(comp_bins[0]))   
                        
        ax.bar(comp_bins[0], comp_bins[1] / comp_bins[1].sum(),
                      color = colors, edgecolor = ['k' for i in range(len(comp_bins[0]))]) 

        #Set axis details
        plt.sca(ax)        
        plt.xlabel(r"$\tau$")
        plt.ylabel(r"$P(\tau$)")
        
    def plot_contrast_complex(self, level_dict_1, level_dict_2, ax = None):        
        """Contrast between two conditions"""
        
        if ax == None:        
            ax = plt.gca()            
        
        #Prep results/ perform contrast            
        X_cond_1 = self.get_levels(level_dict_1) #Get the condition of the levels 
        X_cond_2 = self.get_levels(level_dict_2) #Get the condition of the levels
        
        cond_samples_1 = self.__multiply_X_beta(X_cond_1) 
        cond_samples_2 = self.__multiply_X_beta(X_cond_2) 
        
        cond_rank_1 = rk.rank(cond_samples_1, axis = -1)
        cond_rank_2 = rk.rank(cond_samples_2, axis = -1)
        
        comp_taus = rk.kendall_tau_dist_vec(cond_rank_1, cond_rank_2)       
#        comp_bins = rk.bin_taus(comp_taus, self.K)    
    
    
        #Create the plot
#        colors = np.repeat('#E4F1FE', len(comp_bins[0]))   
#        pdb.set_trace()
        ax.hist(comp_taus, color = '#E4F1FE', density = True)
        sns.kdeplot(comp_taus, ax = ax, color = '#22313F')
#        ax.bar(comp_bins[0], comp_bins[1] / comp_bins[1].sum(),
#                      color = colors, edgecolor = ['k' for i in range(len(comp_bins[0]))]) 

        #Set axis details
        plt.sca(ax)        
        plt.xlabel(r"$\tau_{kendall}$")
        plt.ylabel(r"$P(\tau_{kendall})$")    

    def plot_mean_tau_agreement_old(self, level_dict, ax = None):
        """This didnt seem to be working so depreciateing.
        
        Leaving code just incase"""
        
        if ax == None:
        
            ax = plt.gca()
            
        #Get the condition ranks
        X_cond = self.get_levels(level_dict) #Get the condition of the levels      
        cond_samples = self.__multiply_X_beta(X_cond)
        cond_rank = rk.rank(cond_samples, axis = -1)
        
        mean_rank = rk.rank(cond_samples.mean(0), axis = -1)
        
        
        #Get the data replications for the condition
        mask = np.all(self.X == X_cond, axis = 1)        
        z_rep_cond = self.z_rank[:, mask]
        y_cond = self.y[mask] #Get the data for that condition
        
        
        data_taus = rk.kendall_tau_dist_vec(y_cond, np.tile(mean_rank, (y_cond.shape[0],1)))
        data_taus_mean = data_taus.mean()
        
        #Calculate all the taus
        all_z_tau = np.empty(z_rep_cond.shape[:2])    
        
        for part in range(z_rep_cond.shape[1]):
            all_z_tau[:, part] = rk.kendall_tau_dist_vec(z_rep_cond[:,part], cond_rank)
            
        #Plot the mean taus
        ax.hist(all_z_tau.mean(1), color = '#FFC0CB', alpha = 0.9, rwidth = 1, density = True,
             edgecolor='k', linewidth=0.5)

        ax.axvline(data_taus_mean)
        print("TAu mean = {}".format(data_taus_mean))                
            
    
    def plot_mean_tau_agreement(self, level_dict, ax = None):
                
        if ax == None:
        
            ax = plt.gca()
            
            
        y_rep = self.generate_reps()
        
        
        #Get the condition replications
        X_cond = self.get_levels(level_dict) #Get the condition of the levels     
        mask = np.all(self.X == X_cond, axis = 1)          
        y_rep_cond = y_rep[:, mask]
        
        
        #Get the condition ranks
        cond_samples = self.__multiply_X_beta(X_cond)
        cond_rank = rk.rank(cond_samples, axis = -1)
        mean_rank = rk.rank(cond_samples.mean(0), axis = -1)
        
        all_z_tau = np.empty(y_rep_cond.shape[:2])    
        
        for part in range(y_rep_cond.shape[1]):
            all_z_tau[:, part] = rk.kendall_tau_dist_vec(y_rep_cond[:,part], cond_rank)
            
                            
        #Plot the mean taus
        ax.hist(all_z_tau.mean(1), color = '#FFC0CB', alpha = 0.9, rwidth = 1, density = True,
             edgecolor='k', linewidth=0.5)


        y_cond = self.y[mask] #Get the data for that condition
        data_taus = rk.kendall_tau_dist_vec(y_cond, np.tile(mean_rank, (y_cond.shape[0],1)))
        data_taus_mean = data_taus.mean()
        
        ax.axvline(data_taus_mean, color = 'r', linestyle = '--', alpha = 0.75)
        ax.axvline(all_z_tau.mean(), color = 'k', alpha = 0.75)
       
        plt.sca(ax)        
        plt.xlabel(r"Mean $\tau_{kendall}$")
        
        
def run_sample(iter = 2000, warmup = None, adapt_delta = 0.999, refresh = 250, temp_fnum = 0):        

    """Sample from the posterior. 
    
    Notes: Unfortunately this cannot be implimented in the PyThurstonian class as it will break multiprocessing. This is because we then need to pickle all of PyThurstonian - which fails as patsy is not pickleble"""
    
    module_dir = os.path.dirname(__file__) #Get the module directory
    stan_model_dir = os.path.join(module_dir, 'stan_model')
    data_file_name = os.path.join(stan_model_dir, "temp.data.R")


    #Format the filename correctly
    data_file_name = data_file_name.replace('\\', '/')
    stan_model_dir = stan_model_dir.replace('\\', '/')

    if warmup == None:
        warmup = iter//2           
    
    #Call the Stan program through subprocess
          
    print("Sampling with Stan. Chain ref: {}".format(temp_fnum))
 
    

    command = ["{}/thurstonian_cov.exe".format(stan_model_dir), "sample", "num_samples={}".format(iter), 
        "num_warmup={}".format(warmup), 
        "adapt", "delta={}".format(adapt_delta),
        "data", "file={}".format(data_file_name), "output", 
        "file={}/temp_sample_file_{}.csv".format(stan_model_dir, temp_fnum), "refresh={}".format(refresh)]
   
    sp.run(command, check = True)  
    
