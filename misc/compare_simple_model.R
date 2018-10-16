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


fit <- stan(file = 'thurstonian_cov.stan', data = stan_data, 
            iter = 10000, chains = 6, control = list(adapt_delta = 0.99, max_treedepth = 15),
            cores = 6,
            refresh = 1000)

fit2 <- stan(file = 'thurstonian_simple.stan', data = stan_data, 
            iter = 10000, chains = 6, control = list(adapt_delta = 0.99, max_treedepth = 15),
            cores = 6,
            refresh = 1000)


print(fit, pars = c("beta", "scale"))

print(fit2, pars = c("beta", "scale"))


#Calculate probability of a match from censor model
z_rep = extract(fit)$z_rep
y_rep = apply(z_rep, c(1,2), rank)
y_rep = aperm(y_rep, c(2,3,1))


count = 0
for (i in 1:dim(y_rep)[1]){
  chose_idx = sample(1:dim(y_rep)[2], 2, replace = FALSE)
  y_rep_sample = y_rep[i,chose_idx,]
  
  if (y_rep_sample[1,1] == y_rep_sample[2,1]){
    count = count + 1
  }
  
}

p_match = count / dim(y_rep)[1]


#Calculate probability of a match from simple model
s2 = as_tibble(as.data.frame(fit2)) %>% select('mu_part[1,1]', 'mu_part[1,2]', 'scale[1]')
phi_x = (s2$`mu_part[1,1]`* s2$`scale[1]` - s2$`mu_part[1,2]`* s2$`scale[1]`) 
prob = pnorm(0, phi_x, 1)
inv_prob = 1 - prob

p_match_2 = prob * prob + inv_prob * inv_prob
