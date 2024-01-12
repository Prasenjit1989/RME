import numpy as np
import random as random
import matplotlib.pyplot as plt

np.random.seed(0)
num_samples = 40
num_transmitters = 10
lambda_val = 0.4 # regularization parameter

# generate random measurement data location
measurement_locs = np.array(sorted(random.sample(range(1, 10*num_samples), num_samples)))
data_scale = 10
measurements = data_scale* np.random.rand(num_samples) 


# Compute pairwise distances between measurement locations
pairwise_distances = np.abs(measurement_locs[:, np.newaxis] - measurement_locs[np.newaxis, :])
sigma = 1.0
# Compute radial basis matrix
K = np.exp(-pairwise_distances**2 /(2*sigma**2))

estimated_coeff = np.dot(np.linalg.inv(K + lambda_val * num_samples * np.identity(num_samples)), measurements.reshape(-1, 1))

# Esitmated power value
power_estimated = np.dot(K, estimated_coeff)


plt.figure(figsize = (8, 6))
plt.plot(measurement_locs, measurements, linestyle = "solid", color = "g", label = 'True Map')
plt.plot(measurement_locs, power_estimated, linestyle="dashdot", color = "r", label = 'Estimated Map')

plt.xlabel("Locations")
plt.ylabel("Power")
plt.title("True vs Estimated Power")
plt.legend(loc = "upper right") 
plt.grid(True)
plt.show()