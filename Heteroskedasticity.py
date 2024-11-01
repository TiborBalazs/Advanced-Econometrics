#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 07:22:43 2024

@author: balazstibor
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Function to generate time series data with AR(1) errors
def generate_data(n, alpha, beta, rho, sigma2):
    x = np.random.normal(0, 1, n)  # Generate x values
    u = np.zeros(n)
    epsilon = np.random.normal(0, np.sqrt(sigma2), n)  # Generate error terms epsilon
    
    # Set initial value for u
    u[0] = np.random.normal(0, np.sqrt(sigma2 / (1 - rho**2)))
    
    # Generate AR(1) error terms
    for t in range(1, n):
        u[t] = rho * u[t-1] + epsilon[t]
    
    # Generate y based on the model
    y = alpha + beta * x + u
    return x, y

# Monte Carlo simulation function for OLS
def monte_carlo_simulation(rho, sigma2, num_simulations, n):
    alpha = 1
    beta = 2
    alpha_ols = np.zeros(num_simulations)
    beta_ols = np.zeros(num_simulations)
    
    for i in range(num_simulations):
        x, y = generate_data(n, alpha, beta, rho, sigma2)
        # Perform OLS estimation
        X = np.vstack([np.ones(n), x]).T  # Add constant term for intercept
        ols_estimates = np.linalg.inv(X.T @ X) @ X.T @ y  # OLS formula
        alpha_ols[i] = ols_estimates[0]  # OLS estimate of alpha (intercept)
        beta_ols[i] = ols_estimates[1]   # OLS estimate of beta (slope)
    
    # Return OLS estimates
    return alpha_ols, beta_ols

# Monte Carlo simulation function for GLS with correction for intercept
def monte_carlo_gls_simulation(rho, sigma2, num_simulations, n):
    alpha = 1
    beta = 2
    alpha_gls = np.zeros(num_simulations)
    beta_gls = np.zeros(num_simulations)
    
    for i in range(num_simulations):
        x, y = generate_data(n, alpha, beta, rho, sigma2)
        
        # Transform the data using the GLS transformation
        x_gls = x[1:] - rho * x[:-1]
        y_gls = y[1:] - rho * y[:-1]
        
        # Perform OLS on transformed data (GLS)
        X_gls = np.vstack([np.ones(len(x_gls)), x_gls]).T
        gls_estimates = np.linalg.inv(X_gls.T @ X_gls) @ X_gls.T @ y_gls
        
        # Correct the intercept by dividing by (1 - rho)
        alpha_gls[i] = gls_estimates[0] / (1 - rho)
        beta_gls[i] = gls_estimates[1]
    
    return alpha_gls, beta_gls

# Parameters
num_simulations = 1000
n_small = 100  # Small sample size
n_large = 1000  # Large sample size for asymptotic comparison
sigma2_rho_04 = 1.68
sigma2_rho_08 = 0.72
rho_weak = 0.4
rho_strong = 0.8

# Run simulations for OLS and GLS (weak autocorrelation, rho = 0.4)
alpha_ols_weak_small, beta_ols_weak_small = monte_carlo_simulation(rho_weak, sigma2_rho_04, num_simulations, n_small)
alpha_ols_weak_large, beta_ols_weak_large = monte_carlo_simulation(rho_weak, sigma2_rho_04, num_simulations, n_large)

alpha_gls_weak_small, beta_gls_weak_small = monte_carlo_gls_simulation(rho_weak, sigma2_rho_04, num_simulations, n_small)
alpha_gls_weak_large, beta_gls_weak_large = monte_carlo_gls_simulation(rho_weak, sigma2_rho_04, num_simulations, n_large)

# Run simulations for OLS and GLS (strong autocorrelation, rho = 0.8)
alpha_ols_strong_small, beta_ols_strong_small = monte_carlo_simulation(rho_strong, sigma2_rho_08, num_simulations, n_small)
alpha_ols_strong_large, beta_ols_strong_large = monte_carlo_simulation(rho_strong, sigma2_rho_08, num_simulations, n_large)

alpha_gls_strong_small, beta_gls_strong_small = monte_carlo_gls_simulation(rho_strong, sigma2_rho_08, num_simulations, n_small)
alpha_gls_strong_large, beta_gls_strong_large = monte_carlo_gls_simulation(rho_strong, sigma2_rho_08, num_simulations, n_large)

# Combine results into DataFrames for OLS and GLS (weak autocorrelation)
results_ols_weak_df = pd.DataFrame({
    'alpha_ols': np.concatenate([alpha_ols_weak_small, alpha_ols_weak_large]),
    'beta_ols': np.concatenate([beta_ols_weak_small, beta_ols_weak_large]),
    'n': np.concatenate([['Small Sample (n=100)'] * num_simulations, ['Large Sample (n=1000)'] * num_simulations]),
    'estimator': ['OLS'] * num_simulations * 2
})

results_gls_weak_df = pd.DataFrame({
    'alpha_gls': np.concatenate([alpha_gls_weak_small, alpha_gls_weak_large]),
    'beta_gls': np.concatenate([beta_gls_weak_small, beta_gls_weak_large]),
    'n': np.concatenate([['Small Sample (n=100)'] * num_simulations, ['Large Sample (n=1000)'] * num_simulations]),
    'estimator': ['GLS'] * num_simulations * 2
})

# Combine results into DataFrames for OLS and GLS (strong autocorrelation)
results_ols_strong_df = pd.DataFrame({
    'alpha_ols': np.concatenate([alpha_ols_strong_small, alpha_ols_strong_large]),
    'beta_ols': np.concatenate([beta_ols_strong_small, beta_ols_strong_large]),
    'n': np.concatenate([['Small Sample (n=100)'] * num_simulations, ['Large Sample (n=1000)'] * num_simulations]),
    'estimator': ['OLS'] * num_simulations * 2
})

results_gls_strong_df = pd.DataFrame({
    'alpha_gls': np.concatenate([alpha_gls_strong_small, alpha_gls_strong_large]),
    'beta_gls': np.concatenate([beta_gls_strong_small, beta_gls_strong_large]),
    'n': np.concatenate([['Small Sample (n=100)'] * num_simulations, ['Large Sample (n=1000)'] * num_simulations]),
    'estimator': ['GLS'] * num_simulations * 2
})

# Ensure numeric conversion
results_ols_weak_df['alpha_ols'] = pd.to_numeric(results_ols_weak_df['alpha_ols'], errors='coerce')
results_ols_weak_df['beta_ols'] = pd.to_numeric(results_ols_weak_df['beta_ols'], errors='coerce')
results_gls_weak_df['alpha_gls'] = pd.to_numeric(results_gls_weak_df['alpha_gls'], errors='coerce')
results_gls_weak_df['beta_gls'] = pd.to_numeric(results_gls_weak_df['beta_gls'], errors='coerce')

results_ols_strong_df['alpha_ols'] = pd.to_numeric(results_ols_strong_df['alpha_ols'], errors='coerce')
results_ols_strong_df['beta_ols'] = pd.to_numeric(results_ols_strong_df['beta_ols'], errors='coerce')
results_gls_strong_df['alpha_gls'] = pd.to_numeric(results_gls_strong_df['alpha_gls'], errors='coerce')
results_gls_strong_df['beta_gls'] = pd.to_numeric(results_gls_strong_df['beta_gls'], errors='coerce')

# Plot for Beta estimates (Weak Autocorrelation)
plt.figure(figsize=(10, 6))

# Small sample (n=100) OLS and GLS (weak autocorrelation)
sns.kdeplot(data=results_ols_weak_df[results_ols_weak_df['n'] == 'Small Sample (n=100)'], x='beta_ols', alpha=0.5, label='OLS (Small Sample)', linestyle='-', color='blue')
sns.kdeplot(data=results_gls_weak_df[results_gls_weak_df['n'] == 'Small Sample (n=100)'], x='beta_gls', alpha=0.5, label='GLS (Small Sample)', linestyle='--', color='green')

# Large sample (n=1000) OLS and GLS (weak autocorrelation)
sns.kdeplot(data=results_ols_weak_df[results_ols_weak_df['n'] == 'Large Sample (n=1000)'], x='beta_ols', alpha=0.5, label='OLS (Large Sample)', linestyle='-', color='red')
sns.kdeplot(data=results_gls_weak_df[results_gls_weak_df['n'] == 'Large Sample (n=1000)'], x='beta_gls', alpha=0.5, label='GLS (Large Sample)', linestyle='--', color='purple')

plt.title("Monte Carlo Simulation of Beta (OLS vs GLS, Weak Autocorrelation)")
plt.xlabel("Estimate of Beta")
plt.ylabel("Density")
plt.legend(title="Estimator and Sample Size", loc='upper right')  # Add legend in the upper right
plt.show()

# Plot for Alpha estimates (Weak Autocorrelation)
plt.figure(figsize=(10, 6))

# Small sample (n=100) OLS and GLS (weak autocorrelation)
sns.kdeplot(data=results_ols_weak_df[results_ols_weak_df['n'] == 'Small Sample (n=100)'], x='alpha_ols', alpha=0.5, label='OLS (Small Sample)', linestyle='-', color='blue')
sns.kdeplot(data=results_gls_weak_df[results_gls_weak_df['n'] == 'Small Sample (n=100)'], x='alpha_gls', alpha=0.5, label='GLS (Small Sample)', linestyle='--', color='green')

# Large sample (n=1000) OLS and GLS (weak autocorrelation)
sns.kdeplot(data=results_ols_weak_df[results_ols_weak_df['n'] == 'Large Sample (n=1000)'], x='alpha_ols', alpha=0.5, label='OLS (Large Sample)', linestyle='-', color='red')
sns.kdeplot(data=results_gls_weak_df[results_gls_weak_df['n'] == 'Large Sample (n=1000)'], x='alpha_gls', alpha=0.5, label='GLS (Large Sample)', linestyle='--', color='purple')

plt.title("Monte Carlo Simulation of Alpha (OLS vs GLS, Weak Autocorrelation)")
plt.xlabel("Estimate of Alpha")
plt.ylabel("Density")
plt.legend(title="Estimator and Sample Size", loc='upper right')
plt.show()

# Plot for Beta estimates (Strong Autocorrelation)
plt.figure(figsize=(10, 6))

# Small sample (n=100) OLS and GLS (strong autocorrelation)
sns.kdeplot(data=results_ols_strong_df[results_ols_strong_df['n'] == 'Small Sample (n=100)'], x='beta_ols', alpha=0.5, label='OLS (Small Sample)', linestyle='-', color='blue')
sns.kdeplot(data=results_gls_strong_df[results_gls_strong_df['n'] == 'Small Sample (n=100)'], x='beta_gls', alpha=0.5, label='GLS (Small Sample)', linestyle='--', color='green')

# Large sample (n=1000) OLS and GLS (strong autocorrelation)
sns.kdeplot(data=results_ols_strong_df[results_ols_strong_df['n'] == 'Large Sample (n=1000)'], x='beta_ols', alpha=0.5, label='OLS (Large Sample)', linestyle='-', color='red')
sns.kdeplot(data=results_gls_strong_df[results_gls_strong_df['n'] == 'Large Sample (n=1000)'], x='beta_gls', alpha=0.5, label='GLS (Large Sample)', linestyle='--', color='purple')

plt.title("Monte Carlo Simulation of Beta (OLS vs GLS, Strong Autocorrelation)")
plt.xlabel("Estimate of Beta")
plt.ylabel("Density")
plt.legend(title="Estimator and Sample Size", loc='upper right')
plt.show()

# Plot for Alpha estimates (Strong Autocorrelation)
plt.figure(figsize=(10, 6))

# Small sample (n=100) OLS and GLS (strong autocorrelation)
sns.kdeplot(data=results_ols_strong_df[results_ols_strong_df['n'] == 'Small Sample (n=100)'], x='alpha_ols', alpha=0.5, label='OLS (Small Sample)', linestyle='-', color='blue')
sns.kdeplot(data=results_gls_strong_df[results_gls_strong_df['n'] == 'Small Sample (n=100)'], x='alpha_gls', alpha=0.5, label='GLS (Small Sample)', linestyle='--', color='green')

# Large sample (n=1000) OLS and GLS (strong autocorrelation)
sns.kdeplot(data=results_ols_strong_df[results_ols_strong_df['n'] == 'Large Sample (n=1000)'], x='alpha_ols', alpha=0.5, label='OLS (Large Sample)', linestyle='-', color='red')
sns.kdeplot(data=results_gls_strong_df[results_gls_strong_df['n'] == 'Large Sample (n=1000)'], x='alpha_gls', alpha=0.5, label='GLS (Large Sample)', linestyle='--', color='purple')

plt.title("Monte Carlo Simulation of Alpha (OLS vs GLS, Strong Autocorrelation)")
plt.xlabel("Estimate of Alpha")
plt.ylabel("Density")
plt.legend(title="Estimator and Sample Size", loc='upper right')
plt.show()

