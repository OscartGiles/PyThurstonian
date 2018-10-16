data { 
	int<lower = 1> N; //The number of data points
	int<lower=1> K;  // number of items being ranked 
	int<lower = 1> C; // The number of covariates
	int<lower = 1> J; //Number of raters

	matrix[N, C] X;
	int<lower = 1> rater[N]; //Index the rater who produced rank i

	real<lower = 0> beta_sd_prior;
	real<lower = 0> scale_sd_prior;
} 

parameters { 
	
} 

model{ 

} 

generated quantities{

	vector[K] z[N];
	int y[N, K];

	matrix[C, K-1] beta_zero; //matrix of differences from the first ranking (fixed at zero)
	vector<lower = 0>[J] scale;
	matrix[N, K] mu_part;	

		
	for (j in 1:J){
		scale[j] = lognormal_rng(0, scale_sd_prior);
	}

	
	for (c in 1:C){
		for (k in 1:K-1){
			beta_zero[c, k] = normal_rng(0, beta_sd_prior);
		}
	}	


	mu_part = append_col(rep_vector(0, N), X * beta_zero);

	
	for (i in 1:N){	

		real y_temp[K];

		for (k in 1:K){			
			z[i, k] = normal_rng(mu_part[i, k] * scale[rater[i]], 1); 
		}
		y_temp = sort_indices_asc(z[i]);

		y[i] = sort_indices_asc(y_temp);
	}
	
}
