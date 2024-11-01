# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:52:40 2024

@author: Jebeto
"""

# Extending the code to plot the distributions of x (age) and y (duration of unemployment) for all sample sizes

import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

# Define the sample sizes
sample_sizes = [50, 250, 1000, 5000]

# True parameter value for simulation
true_parameter = 0.025

# Define the number of Monte Carlo simulations
num_simulations = 1000

# Function to simulate data
def simulate_data(n, parameter):
    x = np.random.uniform(18, 60, size=n)  # Simulating 'age' values between 18 and 60
    theta = np.exp(parameter * x)  # Theta depends on age (x)
    y = np.random.exponential(1/theta)  # Simulating 'duration' based on theta
    return x, y

# Log-likelihood function to minimize
def log_likelihood(parameter, x, y):
    theta = np.exp(parameter * x)
    return -np.sum(np.log(theta) - theta * y)

# Function to run the Monte Carlo simulation
def monte_carlo_simulation(n, true_parameter, num_simulations):
    parameter_estimates = []
    x_sample, y_sample = [], []  # To store one sample of x and y for plotting

    for _ in range(num_simulations):
        # Simulate data
        x, y = simulate_data(n, true_parameter)
        if _ == 0:  # Store the first sample for plotting
            x_sample, y_sample = x, y
        
        # Minimize the log-likelihood function to estimate the parameter
        result = opt.minimize(log_likelihood, [0], args=(x, y), method='BFGS')
        parameter_hat = result.x[0]
        
        parameter_estimates.append(parameter_hat)
    
    return np.array(parameter_estimates), x_sample, y_sample

# Store results for each sample size
results = {}

for n in sample_sizes:
    parameter_estimates, x_sample, y_sample = monte_carlo_simulation(n, true_parameter, num_simulations)
    results[n] = {
        "parameter_estimates": parameter_estimates,
        "x_sample": x_sample,
        "y_sample": y_sample,
    }

# Function to plot results of parameter estimates
def plot_results(results, true_parameter):
    plt.figure(figsize=(10, 6))

    for n in sample_sizes:
        parameter_estimates = results[n]['parameter_estimates']
        
        # Plot parameter estimates
        plt.hist(parameter_estimates, bins=30, alpha=0.5, label=f'n = {n}')
        plt.axvline(true_parameter, color='k', linestyle='dashed', linewidth=1)
        plt.title('Distribution of Parameter Estimates')
        plt.xlabel('Parameter')
        plt.ylabel('Frequency')

    plt.legend()
    plt.tight_layout()
    plt.show()

# Function to plot x (age) and y (duration) distributions for all sample sizes
def plot_distributions_all(results):
    for n in sample_sizes:
        x_sample = results[n]["x_sample"]
        y_sample = results[n]["y_sample"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Plot distribution of x (age)
        ax1.hist(x_sample, bins=30, color='blue', alpha=0.7)
        ax1.set_title(f'Distribution of Age (x) for n = {n}')
        ax1.set_xlabel('Age')
        ax1.set_ylabel('Frequency')

        # Plot distribution of y (duration of unemployment)
        ax2.hist(y_sample, bins=30, color='green', alpha=0.7)
        ax2.set_title(f'Distribution of Duration (y) for n = {n}')
        ax2.set_xlabel('Duration of Unemployment')
        ax2.set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()

# Plot the simulation results for parameter estimates
plot_results(results, true_parameter)

# Plot the distributions of x (age) and y (duration) for all sample sizes
plot_distributions_all(results)
