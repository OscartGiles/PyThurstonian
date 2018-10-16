data { 
	int<lower = 1> N; //The number of data points
	int<lower=1> K;  // number of items being ranked 
	int<lower = 1> C; // The number of covariates
	int<lower = 1> J; //Number of raters

	int<lower=1,upper=K> y[N,K];   // y[i] is a vector of rankings for the ith data point
	matrix[N, C] X;
	int<lower = 1> rater[N]; //Index the rater who produced rank i

	real<lower = 0> beta_sd_prior;
	real<lower = 0> scale_sd_prior;
} 

transformed data{
	int y_argsort[N, K];	
	int sort_y_argsort[N, K];

	for (i in 1:N){		
		y_argsort[i] = sort_indices_asc(y[i]);
		sort_y_argsort[i] = sort_indices_asc(y_argsort[i]);
	}

	
}

parameters { 
	ordered[K] z_hat[N];
	matrix[C, K-1] beta_zero; //matrix of differences from the first ranking (fixed at zero)
	vector<lower = 0>[J] scale;
} 

transformed parameters{	
	matrix[N, K] mu_part;	
	mu_part = append_col(rep_vector(0, N), X * beta_zero);	
}

model{ 
	scale ~ lognormal(0, scale_sd_prior);
	to_vector(beta_zero) ~ normal(0, beta_sd_prior);

	for (i in 1:N){					
		z_hat[i] ~ normal(mu_part[i, y_argsort[i]] * scale[rater[i]], 1); 
	}
	
} 

generated quantities{

	matrix[C, K] beta; 	
	matrix[N, K] z;

	matrix[N, K] z_rep;


	//Prepend zeros to beta_zero
	beta = append_col(rep_vector(0.0, C), beta_zero);

	//Recover z

	for (i in 1:N){
		for (k in 1:K){
			z[i, k] = z_hat[i, sort_y_argsort[i, k]];

			z_rep[i,k] = normal_rng(mu_part[i, k] * scale[rater[i]], 1); 
		}	
	}




	
}
