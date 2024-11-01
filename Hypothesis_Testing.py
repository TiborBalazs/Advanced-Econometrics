# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:21:26 2024

@author: Jebeto
"""

import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2
import matplotlib.pyplot as plt

# Set parameters
n_simulations = 1000  # Number of simulations
alpha = 0  # True alpha
beta_true = 0  # True beta under H0 (null hypothesis)
sigma_u = 5  # Range for U(-5, 5) uniform distribution
significance_level = 0.10  # 10% significance level

# Critical values for Wald and LR tests at 10% significance level
critical_value_wald = chi2.ppf(1 - significance_level, df=1)
critical_value_lr = chi2.ppf(1 - significance_level, df=1)

# Sample size for n = 50
n = 50

# Storage for Wald and LR test results for Tibor Balazs
wald_stats = []
lr_stats = []

# Monte Carlo simulation for n = 50
for sim in range(n_simulations):
    # Generate xi ~ N(0, 1) and ui ~ U(-5, 5)
    x = np.random.normal(0, 1, n)
    u = np.random.uniform(-sigma_u, sigma_u, n)
    
    # Generate yi = alpha + beta_true * xi + ui (with beta_true = 0)
    y = alpha + beta_true * x + u
    
    # Perform OLS regression
    X = sm.add_constant(x)  # Add intercept
    model = sm.OLS(y, X).fit()
    
    # Extract beta estimate and its standard error
    beta_hat = model.params[1]
    se_beta_hat = model.bse[1]
    
    # Wald test statistic using the correct formula
    wald_stat = (beta_hat**2) / se_beta_hat**2
    wald_stats.append(wald_stat)
    
    # Total sum of squares (TSS) and residual sum of squares (RSS)
    TSS = np.sum((y - np.mean(y)) ** 2)
    RSS = np.sum(model.resid ** 2)
    
    # LR test statistic using the correct formula
    lr_stat = n * (TSS - RSS) / RSS
    lr_stats.append(lr_stat)

# Convert lists to numpy arrays for easier manipulation
wald_stats = np.array(wald_stats)
lr_stats = np.array(lr_stats)

# Plot distributions of the Wald and LR test statistics
plt.figure(figsize=(12, 6))

# Histogram for Wald test statistics
plt.subplot(1, 2, 1)
plt.hist(wald_stats, bins=30, color='lightblue', edgecolor='black', density=True)
plt.axvline(critical_value_wald, color='red', linestyle='--', label='Critical value (10%)')
plt.title('Distribution of Wald Test Statistics (n=50)')
plt.xlabel('Wald Statistic')
plt.ylabel('Density')
plt.legend()

# Histogram for LR test statistics
plt.subplot(1, 2, 2)
plt.hist(lr_stats, bins=30, color='lightgreen', edgecolor='black', density=True)
plt.axvline(critical_value_lr, color='red', linestyle='--', label='Critical value (10%)')
plt.title('Distribution of LR Test Statistics (n=50)')
plt.xlabel('LR Statistic')
plt.ylabel('Density')
plt.legend()

# Show plot
plt.tight_layout()
plt.show()

# Calculate the empirical size distortion for n=50
wald_rejections = np.sum(wald_stats > critical_value_wald) / n_simulations
lr_rejections = np.sum(lr_stats > critical_value_lr) / n_simulations

# Output the empirical size distortion
print(f"Empirical size distortion (Wald test, n=50): {wald_rejections:.3f}")
print(f"Empirical size distortion (LR test, n=50): {lr_rejections:.3f}")

# Set a range of sample sizes to analyze size distortion
sample_sizes = [20, 50, 100, 200, 500]
wald_size_distortions = []
lr_size_distortions = []

# Monte Carlo simulation for different sample sizes
for n in sample_sizes:
    wald_rejections = 0
    lr_rejections = 0
    for sim in range(n_simulations):
        # Generate xi ~ N(0, 1) and ui ~ U(-5, 5)
        x = np.random.normal(0, 1, n)
        u = np.random.uniform(-sigma_u, sigma_u, n)

        # Generate yi = alpha + beta_true * xi + ui (with beta_true = 0)
        y = alpha + beta_true * x + u

        # Perform OLS regression
        X = sm.add_constant(x)  # Add intercept
        model = sm.OLS(y, X).fit()

        # Extract beta estimate and its standard error
        beta_hat = model.params[1]
        se_beta_hat = model.bse[1]

        # Wald test statistic
        wald_stat = (beta_hat**2) / se_beta_hat**2

        # Total sum of squares (TSS) and residual sum of squares (RSS)
        TSS = np.sum((y - np.mean(y)) ** 2)
        RSS = np.sum(model.resid ** 2)

        # LR test statistic
        lr_stat = n * (TSS - RSS) / RSS

        # Check for rejection of null hypothesis
        if wald_stat > critical_value_wald:
            wald_rejections += 1
        if lr_stat > critical_value_lr:
            lr_rejections += 1

    # Calculate the empirical size distortion for each sample size
    wald_size_distortions.append(wald_rejections / n_simulations)
    lr_size_distortions.append(lr_rejections / n_simulations)

# Plot the empirical size distortions for different sample sizes
plt.figure(figsize=(8, 6))
plt.plot(sample_sizes, wald_size_distortions, label='Wald Test', marker='o')
plt.plot(sample_sizes, lr_size_distortions, label='LR Test', marker='s')
plt.axhline(y=0.10, color='r', linestyle='--', label='10% Nominal Size')
plt.title('Empirical Size Distortion for Different Sample Sizes')
plt.xlabel('Sample Size')
plt.ylabel('Empirical Size Distortion')
plt.legend()
plt.grid(True)
plt.show()

# Output the empirical size distortion for different sample sizes
for i, n in enumerate(sample_sizes):
    print(f"Sample size: {n}, Empirical size distortion (Wald): {wald_size_distortions[i]:.3f}, Empirical size distortion (LR): {lr_size_distortions[i]:.3f}")
