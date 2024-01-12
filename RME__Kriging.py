import numpy as np
import matplotlib.pyplot as plt
import random

np.random.seed(0)
num_samples = 1000
num_transmitters = 1

p_TX = 10 # Transmitted power by the transmitter

random.seed(2)

# locations of the transmitters
transmitters_locs = np.array(sorted(random.sample(range(1, 5*num_samples), num_transmitters)))

# Generating measurements at known transmitter locations
measurement_locs = np.array(sorted(random.sample(range(1, 5*num_samples), num_samples)))
# measurement_locs = np.linspace(0, 10, num_samples).reshape(num_samples, 1)

# Creating measurement matrix for holding measurement data from different transmitters
measurements = np.zeros ((num_samples, num_transmitters))
# distances = np.abs(np.tile(measurement_locs[:, np.newaxis], (1, num_transmitters)) - 
#                    np.tile(transmitters_locs[np.newaxis, :], (num_samples, 1)))
# # Generating measurement data from true coefficients and distance values
# A = 1.0 / (distances + 0.1) # Psi matrix in the paper 

# Generating measurement data from true coefficients and distance values 
measurements = np.random.rand(num_samples) 

mu_p = p_TX + h_PL - mu_SF - mu_FF