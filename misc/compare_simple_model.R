library(tidyverse)
library(rstan)
library(bayesplot)


data = read_csv("rank_data.csv")


stan_data = list(N = dim(data)[1], K = 2, C = 1, J = 1,
                 y = data.matrix(data %>% select(Y1, Y2)),
                 X = as.matrix(rep(1, dim(data)[1])),
                 rater = rep(1, dim(data)[1]),
                 beta_sd_prior = 2 * 3,
                 scale_sd_prior = 0.5)


iterations = 10000

fit1 <- stan(file = 'thurstonian_cov.stan', data = stan_data, 
            iter = iterations, chains = 6, control = list(adapt_delta = 0.99, max_treedepth = 20),
            cores = 6,
            refresh = 1000)

fit2 <- stan(file = 'thurstonian_cov_sigma.stan', data = stan_data, 
             iter = iterations, chains = 6, control = list(adapt_delta = 0.99, max_treedepth = 20),
             cores = 6,
             refresh = 1000)


fit3 <- stan(file = 'thurstonian_simple.stan', data = stan_data, 
            iter = iterations, chains = 6, control = list(adapt_delta = 0.99, max_treedepth = 20),
            cores = 6,
            refresh = 1000)


s1 = as.data.frame(fit1)
s2 = as.data.frame(fit2)
s3 = as.data.frame(fit3)

sigma1 = s1$`sigma[1]`
sigma2 = s2$`sigma[1]`
sigma3 = s3$`sigma[1]`


samps = tibble(s1 = s1$`beta_zero[1,1]`, 
               s2 = s2$`beta_zero[1,1]`, 
               s3 = s3$`beta_zero[1,1]`,
               sigma1 = sigma1, sigma2 = sigma2, sigma3 = sigma3)


ggplot(data = samps) + 
  geom_density(aes(x= s1), alpha = 0.01, color = 'red') +
  geom_density(aes(x= s2), alpha = 0.01, color = 'blue') +
  geom_density(aes(x= s3), alpha = 0.01, color = 'green') 

ggplot(data = samps) + 
  geom_density(aes(x= sigma1), alpha = 0.01, color = 'red') +
  geom_density(aes(x= sigma2), alpha = 0.01, color = 'blue') +
  geom_density(aes(x= sigma3), alpha = 0.01, color = 'green') 




print(fit1, pars = c("beta_zero", "sigma"), digits = 4)
print(fit2, pars = c("beta_zero", "sigma"), digits = 4)
print(fit3, pars = c("beta_zero", "sigma"), digits = 4)

assert_match = function(mu1, mu2, sigma){
  
  a = (mu1 - mu2) / sqrt(2 * sigma^2)
  
  tau = 1 / sigma
  
  b = ((mu1 - mu2) * tau) / sqrt(2)
  
  return(c(a, b))
  
}

mu1 = 0
mu2 = 4.2
sigma = 2.0
tau = 1/ sigma

N = 100000
d1 = tibble(z1 = rnorm(N, mu1, sigma), z2 = rnorm(N, mu2, sigma))
d2 = tibble(z1 = rnorm(N, mu1 * tau, 1), z2 = rnorm(N, mu2 * tau, 1))

# ggplot() + 
#     geom_histogram(data = d1, aes(x = z1, y = ..density..), alpha = 0.2, fill = "red") +
#       geom_histogram(data = d2, aes(x = z1, y = ..density..), alpha = 0.2, fill = "blue")
# 
# 
# ggplot() + 
#   geom_histogram(data = d1, aes(x = z2, y = ..density..), alpha = 0.2, fill = "red") +
#   geom_histogram(data = d2, aes(x = z2, y = ..density..), alpha = 0.2, fill = "blue")


d1 %>% summarise(sum(z1 < z2) / n())
d2 %>% summarise(sum(z1 < z2) / n())
