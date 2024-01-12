#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:03:18 2023

@author: prasenjitd
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
num_samples = 100
num_transmitters = 10


# locations of the transmitters
transmitters_locs = np.linspace(0, 10, num_transmitters).reshape(1, num_transmitters)

# Generating measurements at known transmitter locations
measurement_locs = np.linspace(0, 10, num_samples).reshape(num_samples, 1)

# True coefficients for transmitters
true_coeffs = np.random.randn(num_transmitters, 1)

# Creating measurement matrix for holding measurement data from different transmitters
measurements = np.zeros ((num_samples, num_transmitters))
distances = np.abs(np.tile(measurement_locs, (1, num_transmitters)) - np.tile(transmitters_locs, (num_samples, 1)))
# Generating measurement data from true coefficients and distance values
A = 1.0 / (distances + 0.1) # Psi matrix in the paper 

# Generating measurement data from true coefficients and distance values and then add noise
noise_level = 0.6
measurements = np.dot(A, true_coeffs) + np.random.randn(num_samples, 1) * noise_level

# Estimating the coefficients for the transmitters
estimated_coeffs = np.zeros((num_transmitters, 1))
estimated_coeffs = np.dot((np.dot(np.linalg.inv(np.dot(A.T, A)), A.T)), measurements)

# Using numpy implemented least square methods. For y = A * x, np.linalg.lstsq(A, y, rcond = None)
alpha = np.linalg.lstsq(A, measurements, rcond = None)[0]


plt.figure(figsize = (8, 6))
plt.plot(transmitters_locs.T, true_coeffs, linestyle = "solid", color = "g")
plt.plot(transmitters_locs.T, estimated_coeffs, linestyle="dashdot", color = "r")

plt.xlabel("Transmitter Locations")
plt.ylabel("Power Coefficients")
plt.title("True vs Estimated Power Coefficients at Transmitter Locations")
plt.grid(True)
plt.show()

measurements_true = np.dot(A, true_coeffs)
measurements_estimation = np.dot(A, estimated_coeffs)

plt.figure(figsize = (8, 6))
plt.plot(measurement_locs, measurements_true, linestyle = "solid", color = "g", label = 'True Map')
plt.plot(measurement_locs, measurements_estimation, linestyle="dashdot", color = "r", label = 'Estimated Map')

plt.xlabel("Locations")
plt.ylabel("Power")
plt.title("True vs Estimated Power")
plt.legend(loc = "upper right") 
plt.grid(True)
plt.show()