#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 12:02:29 2024

@author: prasenjitd
"""

import numpy as np
import matplotlib.pyplot as plt
import random


def Polynomial_RME_estimation(measurements, measurement_locs, grid_x_test, degree = 3):
    
    # Construct the polynomial features
    X_poly = np.vander(measurement_locs, degree + 1, increasing = True)
    
    # Perform linear regression
    estimated_coeff = np.linalg.lstsq(X_poly, measurements, rcond = None)[0]
    
    # construct the polynomial features for the grid
    grid_x_poly = np.vander(grid_x_test, degree+1, increasing = True)
    
    # Predicted power
    predicted_power = np.dot(grid_x_poly, estimated_coeff)
    
    return predicted_power

def linear_interpolation_RME_estimation(measurements, measurement_locs, grid_x_test):
    
    predicted_power = np.zeros(grid_x_test.shape)
    for i in range(len(grid_x_test)):
        k = np.searchsorted(measurement_locs, grid_x_test[i], side = 'right')
        if k == 0:
            predicted_power[i] = measurements[0]
        elif k == len(measurements):
            predicted_power[i] = measurements[-1]
        else:
            x_left, x_right = measurement_locs[k-1], measurement_locs[k]
            y_left, y_right = measurements[k-1], measurements[k]
            predicted_power[i] = y_left + (grid_x_test[i] - x_left)*(y_right - y_left)/(x_right - x_left)
    
    return predicted_power
            
def KNN_RME_estimation(measurements, measurement_locs, grid_x_test, K_neighbors = 5):
    from sklearn.neighbors import KNeighborsRegressor
    X_train, X_test = measurement_locs.reshape(-1, 1), grid_x_test.reshape(-1, 1)
    knn_reg = KNeighborsRegressor(n_neighbors= K_neighbors, weights='distance')
    knn_reg.fit(X_train, measurements)
    
    predicted_power = knn_reg.predict(X_test)
    return predicted_power

def DNN_RME_estimation(measurements, measurement_locs, grid_x_test):
    import tensorflow as tf
    X_train, X_test = measurement_locs.reshape(-1, 1), grid_x_test.reshape(-1, 1)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation = 'relu', input_shape = (X_train.shape[1], )),
        tf.keras.layers.Dense(16, activation = 'relu'),
        tf.keras.layers.Dense(1)])
    
    model.compile(optimizer = 'adam', loss = 'mse')
    model.fit(X_train, measurements, epochs = 50, batch_size = 16)
    predicted_power = model.predict(X_test)
    
    return predicted_power


alpha_gain = 20 # transmitted power * (lambda/(4*pi))**2
r_x_min, r_x_max = 1, 200
num_samples = 80
num_transmitters = 1
random.seed(2)

# locations of the transmitters
transmitters_locs, beta_s = np.array([round(random.uniform(r_x_min + 10, r_x_max - 10), 3) 
                                      for _ in range(num_transmitters)]).reshape(1, -1), np.array(random.uniform(0, 1))

# Generating measurements at known transmitter locations
measurement_locs = np.array(sorted(np.round(np.random.uniform(r_x_min, r_x_max, num_samples), 2)))

# Creating measurement matrix for holding measurement data from different transmitters
measurements = np.zeros ((num_samples, num_transmitters))
distances = np.abs(np.tile(measurement_locs[:, np.newaxis], (1, num_transmitters)) - 
                    np.tile(transmitters_locs, (num_samples, 1)))

# Generating measurement data from true coefficients and distance values
noise_level = 0.01
measurements = alpha_gain / (distances + beta_s**2) + np.random.randn(num_samples, 1) * noise_level
measurements_log = 10*np.log10(measurements)

# Create a grid for prediction    
grid_size = 300
grid_x_test = np.linspace(r_x_min, r_x_max, grid_size)
degree = 5
estimated_power_poly = Polynomial_RME_estimation(measurements_log, measurement_locs, grid_x_test, degree= degree)
estimated_power_interp = linear_interpolation_RME_estimation(measurements_log, measurement_locs, grid_x_test)
estimated_power_knn = KNN_RME_estimation(measurements_log, measurement_locs, grid_x_test, K_neighbors = 5)
estimated_power_dnn = DNN_RME_estimation(measurements_log, measurement_locs, grid_x_test)


plt.figure(figsize = (8, 6))
plt.plot(measurement_locs, measurements_log, linestyle = "solid", color = "g", label = 'True Map')
plt.plot(grid_x_test, estimated_power_poly, linestyle="dashdot", color = "r", label = 'Estimated Map (Polynomial)')
plt.plot(grid_x_test, estimated_power_interp, linestyle="dashdot", color = "m", label = 'Estimated Map (Interpolation)')
plt.plot(grid_x_test, estimated_power_knn, linestyle="dashdot", color = "y", label = 'Estimated Map (KNN)')
plt.plot(grid_x_test, estimated_power_dnn, linestyle="dashdot", color = "c", label = 'Estimated Map (DNN)')

plt.xlabel("Locations")
plt.ylabel("Power (dB)")
plt.title("True vs Estimated Power")
plt.legend(loc = "upper right") 
plt.grid(True)
plt.show()
