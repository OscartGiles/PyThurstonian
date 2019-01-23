# PyThurstonian
A python module for fitting Thurstonian models using Stan. 

This is a pre-release version. Use entirely at your own risk. 

## Install
PyThurstonian has a number of depencies, all of which are part of the standard Anaconda Python distribution (https://www.anaconda.com/download/). The easiest way to work with PyThurstonian is to download Anaconda. Note that you will need to use the **64 bit version** of Anaconda and PyThurstonian **is only supported on Windows** at the moment (Linux release also planned). 

You can then install PyThurstonian using pip (read the uninstall instructions first). Simple open a command window (or Anaconda Prompt if anaconda is not on your system path) and type:

```
pip install git+https://github.com/OscartGiles/PyThurstonian_public
```

To check everything has worked grab the  simulated_data_example.py program in Examples/ folder in this repo. Run the program and it should sample from a Thurstonian model with a simulated dataset. 

## Uninstall
To uninstall simply run: 

```
pip uninstall PyThurstonian
```

However, **if you have run model fits it may not remove all files from your anaconda directory** (this is a bug which will soon be fixed). After uninstalling through pip you may need to manually remove the PyThurstonian folder in your site packages. 

## Worked example
The Examples_markdown folder contains an example as an html file. 

## Model validation

The Thurstonian model's likelihood function can be evaluated analytically when only two items are being ranked. My implimentation uses a "rank censoring" trick to obviate the need to analtically evaluate the likelihood function, which allows us to extend the model to many more items. However, I implimented both the analytic and "rank censored" models and compare both with datasets of only 2 items. This was done to ensure that the "rank censored" model recovers the correct posterior. This can be replicated by running the "compare_simple_model.R" file in the misc folder. The models recover the same parameters, although the rank censored model is a little slower and has a smaller effective sample size. 
