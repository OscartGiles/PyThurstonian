library(rstan)
library(tidyverse)
library(shinystan)

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

x = as.matrix(read_csv("X.csv"))
y = as.matrix(read_csv("y.csv"))
subj = read_csv("subj.csv")
subj$Subj = as.numeric(subj$Subj)

stan_data = list(N = dim(y)[1], K = dim(y)[2], C = 2, J = dim(unique(subj))[1], X = x, y = y, 
              rater = subj$Subj, beta_sd_prior = 3, scale_sd_prior = 0.5)


fit <- stan(file = 'thurstonian_cov.stan', data = stan_data, iter = 10000, chains = 4, control = list(adapt_delta = 0.99))

launch_shinystan(fit)
