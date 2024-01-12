import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import random
import matplotlib.pyplot as plt

min_range, max_range = 0, 100
num_samples = 1000

# Transmitter location
transmitter_loc = np.array([max_range*random.random(), max_range*random.random()]) # random.ranodm() generates data b/w 0 and 1
true_coeff = 10

# Simulated terrain map
terrain_map = np.random.rand(max_range, max_range)

# Generate sample points in 2D space for the environment
x = np.random.uniform(min_range, max_range, num_samples)
y = np.random.uniform(min_range, max_range, num_samples)

# Distances between tranmitter and measurement points
distances = np.sqrt((x-transmitter_loc[0])**2 + (y-transmitter_loc[1])**2)

# Measurement data
measurements = true_coeff / (distances + 1) + np.random.rand(num_samples)

# Prepare data for training
input_data = np.zeros((num_samples, terrain_map.shape[0], terrain_map.shape[1], 2))
# meas_point_mask = np.zeros((max_range, max_range))
# meas_point_mask[np.round(x).astype(int), np.round(y).astype(int)] =1
for i in range(num_samples):
    input_data[i, :, :, 0] = terrain_map  # Terrain map as input
    input_data[i, int(x[i]), int(y[i]), 1] = 1  # Coordinates as additional input
output_data = measurements.reshape(-1, 1)

# Define a CNN model 
input_shape = input_data.shape[1:]
model = tf.keras.Sequential([
    layers.Conv2D(16, (3, 3), activation = 'relu', input_shape = input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation = 'relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation = 'relu'),
    layers.Dense(1)
    ]) 
model.compile(optimizer = 'adam', loss = 'mse')
# train the model 
model.fit(input_data, output_data, epochs = 30, batch_size = 16)

# Generate grid points for prediction
grid_size = 100
xx, yy = np.meshgrid(np.linspace(min_range, max_range, grid_size, endpoint= False), 
                     np.linspace(min_range, max_range, grid_size, endpoint= False))

grid_pts_val = np.stack((xx.ravel(), yy.ravel())).T

grid_points = np.zeros((grid_size*grid_size, terrain_map.shape[0], terrain_map.shape[1], 2))

for k in range(grid_size*grid_size):    
    grid_points[k, :, :, 0] = terrain_map
    grid_points[k, int(grid_pts_val[k, 0]), int(grid_pts_val[k, 1]), 1] = 1

predicted_power = model.predict(grid_points).reshape(grid_size, grid_size)

plt.figure(figsize = (8, 6))
plt.imshow(predicted_power, extent = (min_range, max_range, min_range, max_range), origin = 'lower')
plt.colorbar(label='Estimated Power')
plt.scatter(transmitter_loc[0], transmitter_loc[1], c='red', marker = 'x', label = 'Transmitter')
plt.xlabel('X - coordinate')
plt.ylabel('Y - coordinate')
plt.title('Estimated Power map using CNN with Terrain Map')
plt.legend()
plt.grid(True)
plt.show()



