# PyThurstonian_public
A public snapshot of the PyThurstonian repository, which will be released later this year. Unfortunately that means issues and commits are not visible. Unfortunately the full repository cannot be released yet as it contains sensitive data. 


# PyThurstonian
A simple python module for fitting Thurstonian models using Stan. 


## Install

### Backend
PyThurstonian requires *CmdStan* (http://mc-stan.org/users/interfaces/cmdstan) to be installed on your machine. To get the latest version go to https://github.com/stan-dev/cmdstan/releases and grab the latest release. An instaltion guide can be found on the same page.


## Examples
The Examples_markdown folder contains an example as an html file. 


## Model validation

The Thurstonian model's likelihood function can be evaluated analytically when only two items are being ranked. My implimentation uses a "rank censoring" trick to obviate the need to analtically evaluate the likelihood function, which allows us to extend the model to many more items. However, I implimented both the analytic and "rank censored" models and compare both with datasets of only 2 items. This was done to ensure that the "rank censored" model recovers the correct posterior. This can be replicated by running the "compare_simple_model.R" file in the misc folder. The models recover the same parameters, although the rank censored model is a little slower and has a smaller effective sample size. 
