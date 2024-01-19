from sklearn.neighbors import KNeighborsRegressor
from map_generator_1d import MapGenerator1D
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import sys

class MapEstimator:
    def __init__(self, v_measurement_locs, v_measurements, v_grid_x):
        self.v_measurement_locs = v_measurement_locs
        self.v_measurements = v_measurements
        self.v_grid_x = v_grid_x
        
    def estimate(self, v_measurement_locs, v_measurements, v_grid_x):
        # write common funcitons
        return self._estimate(v_measurement_locs, v_measurements, v_grid_x)
    
    def _estimate(self, v_measurement_locs, v_measurements, v_grid_x):
        raise NotImplementedError


class PolynomialEstimator(MapEstimator):
    def __init__(self, v_measurement_locs, v_measurements, v_grid_x, degree=5):
        super().__init__(v_measurement_locs, v_measurements, v_grid_x)
        self.degree = degree
    
    def estimate(self):
        v_meas_locs_poly = np.vander(self.v_measurement_locs, self.degree +1, increasing = True)
        estimated_coeff = np.linalg.lstsq(v_meas_locs_poly, self.v_measurements, rcond = None)[0]
        v_grid_x_poly = np.vander(self.v_grid_x, self.degree + 1, increasing = True)
        return np.dot(v_grid_x_poly, estimated_coeff)


class LinearInterpEstimator(MapEstimator):
    def __init__(self, v_measurement_locs, v_measurements, v_grid_x):
        super().__init__(v_measurement_locs, v_measurements, v_grid_x)
    
    def estimate(self):
        estimated_power = np.zeros(self.v_grid_x.shape)
        for i in range(len(self.v_grid_x)):
            k = np.searchsorted(self.v_measurement_locs, self.v_grid_x[i], side = 'right')
            
            if k == 0:
                estimated_power[i] = self.v_measurements[0]
            elif k == len(self.v_measurements):
                estimated_power[i] = self.v_measurements[-1]
            else:
                x_left, x_right = self.v_measurement_locs[k-1], self.v_measurement_locs[k]
                y_left, y_right = self.v_measurements[k-1], self.v_measurements[k]
                estimated_power[i] = y_left + (self.v_grid_x[i] - x_left)*(y_right - y_left)/(x_right - x_left)        
        return estimated_power
    

class KnnEstimator(MapEstimator):
    def __init__(self, v_measurement_locs, v_measurements, v_grid_x, num_neighbors=5):
        super().__init__(v_measurement_locs, v_measurements, v_grid_x)
        self.num_neighbors = num_neighbors
        self.knn_reg = KNeighborsRegressor(n_neighbors=self.num_neighbors, weights='distance')

    def estimate(self):
        self.knn_reg.fit(self.v_measurement_locs.reshape(-1, 1), self.v_measurements.reshape(-1, 1))
        return self.knn_reg.predict(self.v_grid_x.reshape(-1, 1)).ravel()
    
class DnnEstimator(MapEstimator):
    def __init__(self, input_layer_size, hidden_layers_size, output_layer_size):
        self.model = self.build_model(input_layer_size, hidden_layers_size, output_layer_size)      
    
    def build_model(self, input_layer_size, hidden_layers_size, output_layer_size):
        model = models.Sequential()
        model.add(layers.Dense(hidden_layers_size[0], activation = 'relu', input_shape = (input_layer_size, ))),
        for i in range(1, len(hidden_layers_size)):
            model.add(layers.Dense(hidden_layers_size[i], activation = 'relu')),
        
        model.add(layers.Dense(output_layer_size))
        model.compile(optimizer = 'adam', loss = 'mse')
        return model            
        
    def train_model(self, X_train, y_train, epochs = 100, batch_size = 32):
        self.model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size)
    
    def save_weights(self, file_name):
        self.model.save_weights(file_name)
        
    def load_weights (self, file_name):
        self.model.load_weights(file_name)
        
    def estimate(self, X_test):        
        return self.model.predict(X_test)        
        


# Initialize parameters
mapGen = MapGenerator1D(
    alpha = 30,
    n_tx = 3,
    r_x_min = 1,
    r_x_max = 100
)

num_points = 100
num_maps = 1
y_coord = 1
num_samples = 20
noise_level = 0.1

m_tx_locs, v_x, v_received_power = mapGen.generate_map(num_points = num_points, y_coord = y_coord)

v_noise = mapGen.generate_noise(noise_level = noise_level, 
                            num_points = num_points)

v_sampling_mask, v_measurement_locs = mapGen.generate_sampling_mask(num_samples = num_samples, 
                                               num_points = num_points)

v_measurements = mapGen.generate_measurement_data(v_x, v_received_power, v_noise, 
                                                                  v_sampling_mask)

# KNN estimation
num_neighbors=5
knn_estimation = KnnEstimator(v_measurement_locs, 
                              v_measurements, 
                              v_x, 
                              num_neighbors=num_neighbors)

v_predicted_power_knn = knn_estimation.estimate()


# Polynomial estimation
degree=5
poly_estimation = PolynomialEstimator(v_measurement_locs, 
                              v_measurements, 
                              v_x, 
                              degree=degree)

v_predicted_power_poly = poly_estimation.estimate()

# Linear Interpolation estimation
lin_interp_estimation = LinearInterpEstimator(v_measurement_locs, v_measurements, v_x)

v_predicted_power_lin_interp = lin_interp_estimation.estimate()


# DNN Estimation

def data_generation_dnn(num_maps=100, num_points = num_points, y_coord = y_coord, num_samples = 20, noise_level = 0.1):

    dict_true_map= {
        'm_tx_locs': [],
        'm_x': [],
        'm_received_power': [],
        'y_coord': []    
        }
    
    m_noise = []

    dict_sampling_mask = {
        'm_sampling_mask': [],
        'm_measurement_locs': []
        }

    m_measurements =[]
    for _ in range(num_maps):    
        m_tx_locs, v_x, v_received_power = mapGen.generate_map(num_points = num_points, y_coord = y_coord)

        dict_true_map['m_tx_locs'].append(m_tx_locs)
        dict_true_map['m_x'].append(v_x)
        dict_true_map['m_received_power'].append(v_received_power)
        dict_true_map['y_coord'].append(y_coord)

        v_noise = mapGen.generate_noise(noise_level = noise_level, 
                                    num_points = num_points)
        m_noise.append(v_noise)
        
        v_sampling_mask, v_measurement_locs = mapGen.generate_sampling_mask(num_samples = num_samples, 
                                                       num_points = num_points)
        
        dict_sampling_mask['m_sampling_mask'].append(v_sampling_mask)
        dict_sampling_mask['m_measurement_locs'].append(v_measurement_locs)    
        
        m_measurements.append(mapGen.generate_measurement_data(v_x, v_received_power, v_noise, 
                                                                          v_sampling_mask))
    # Convert list into array in dictionary
    dict_true_map['m_tx_locs'] = np.array(dict_true_map['m_tx_locs'])
    dict_true_map['m_x'] = np.array(dict_true_map['m_x'])
    dict_true_map['m_received_power'] = np.array(dict_true_map['m_received_power'])
    dict_true_map['y_coord'] = np.array(dict_true_map['y_coord'])

    m_noise = np.array(m_noise)

    dict_sampling_mask['m_sampling_mask'] = np.array(dict_sampling_mask['m_sampling_mask'])
    dict_sampling_mask['m_measurement_locs'] = np.array(dict_sampling_mask['m_measurement_locs'])

    m_measurements = np.array(m_measurements)
    
    return dict_true_map, m_noise, dict_sampling_mask, dict_sampling_mask['m_measurement_locs'], m_measurements


dict_true_map_train, _, _, m_measurement_locs_train, m_measurements_train = data_generation_dnn(num_maps=1000,
                                            num_points = num_points, y_coord = y_coord, num_samples = 20, noise_level = 0.1)
dict_true_map_test, _, _, m_measurement_locs_test, m_measurements_test = data_generation_dnn(num_maps=30, 
                                            num_points = num_points, y_coord = y_coord, num_samples = 20, noise_level = 0.1)

# Training and test data
X_train = np.concatenate((m_measurement_locs_train, m_measurements_train), axis = 1)
y_train = dict_true_map_train['m_received_power']
X_test = np.concatenate ((m_measurement_locs_test, m_measurements_test), axis = 1)

# Normalize data
X_train_norm = X_train / X_train.max(axis = 0)
X_test_norm = X_test / X_test.max(axis = 0)

# DNN size
input_layer_size = X_train.shape[1]
hidden_layers_size = [128, 64]
output_layer_size = y_train.shape[1]

dnn_model = DnnEstimator(input_layer_size = input_layer_size, 
                             hidden_layers_size = hidden_layers_size,
                             output_layer_size = output_layer_size
                             )
# train a model
dnn_model.train_model(X_train_norm, y_train, epochs = 500, batch_size = 32)

# Save weights to a file
weight_file_name = 'rme_estimation_1d_dnn_weights.h5'
dnn_model.save_weights(weight_file_name)

# Testing a model
new_dnn_model = DnnEstimator(input_layer_size = input_layer_size, 
                             hidden_layers_size = hidden_layers_size,
                             output_layer_size = output_layer_size)
new_dnn_model.load_weights(weight_file_name)

# Estimate map
m_estimated_maps = new_dnn_model.estimate(X_test_norm)
 