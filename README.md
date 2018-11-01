# PyThurstonian
A simple python module for fitting Thurstonian models using Stan. 


## Install
You can install the latest development version of PyThurstonian using pip (maybe unstable). Offical releases will be available soon.

```
pip install https://github.com/OscartGiles/PyThurstonian
```

To check everything has worked run the simulated_data_example.py program in Examples/ folder. The program should sample from a simple Thurstonian model for a simulated dataset. 

## Worked example
The Examples_markdown folder contains an example as an html file. 


## Model validation

The Thurstonian model's likelihood function can be evaluated analytically when only two items are being ranked. My implimentation uses a "rank censoring" trick to obviate the need to analtically evaluate the likelihood function, which allows us to extend the model to many more items. However, I implimented both the analytic and "rank censored" models and compare both with datasets of only 2 items. This was done to ensure that the "rank censored" model recovers the correct posterior. This can be replicated by running the "compare_simple_model.R" file in the misc folder. The models recover the same parameters, although the rank censored model is a little slower and has a smaller effective sample size. 
