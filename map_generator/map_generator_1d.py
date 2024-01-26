import random
import numpy as np


class MapGenerator1D:
    def __init__(self, alpha, num_tx, x_lim, y_lim, y_coord_meas_line):
        self.alpha = alpha
        self.num_tx = num_tx        
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.y_coord_meas_line = y_coord_meas_line
    
    def generate_tx_locations(self):
        return np.array([[random.uniform(self.x_lim[0], self.x_lim[1]), random.uniform(self.y_lim[0], self.y_lim[1])]
            for _ in range(self.num_tx)])
        
    def generate_map(self, num_points):
        m_tx_locs = self.generate_tx_locations()
        v_x = np.linspace(self.x_lim[0], self.x_lim[1], num_points)
        m_x_coord_distances = np.tile(v_x, (self.num_tx, 1)) - np.tile(m_tx_locs[:, 0].reshape(-1, 1), (1, v_x.shape[0]))
        m_betas = np.tile(m_tx_locs[:, 1].reshape(-1, 1), (1, v_x.shape[0])) - self.y_coord_meas_line
        v_true_pow = np.sum(self.alpha/ (m_x_coord_distances**2 + m_betas**2), axis = 0)        
        
        return v_x, v_true_pow      
     

""" def data_generation_dnn(mapGen, num_maps=100, num_points = 100, y_coord = 1, num_samples = 20, noise_level = 0.1):
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
    mapGen = MapGenerator1D()
    for _ in range(num_maps):    
        m_tx_locs, v_x, v_true_power = mapGen.generate_map(num_points = num_points, y_coord = y_coord)

        dict_true_map['m_tx_locs'].append(m_tx_locs)
        dict_true_map['m_x'].append(v_x)
        dict_true_map['m_received_power'].append(v_true_power)
        dict_true_map['y_coord'].append(y_coord)

        v_noise = mapGen.generate_noise(noise_level = noise_level, 
                                    num_points = num_points)
        m_noise.append(v_noise)
        
        v_sampling_mask, v_measurement_locs = mapGen.generate_sampling_mask(num_meas = num_samples, 
                                                       num_points = num_points)
        
        dict_sampling_mask['m_sampling_mask'].append(v_sampling_mask)
        dict_sampling_mask['m_measurement_locs'].append(v_measurement_locs)    
        
        m_measurements.append(mapGen.generate_measurement_data(v_x, v_true_power, v_noise, 
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
    
    return dict_true_map['m_received_power'], dict_sampling_mask['m_measurement_locs'], m_measurements, m_noise, dict_sampling_mask
 """
