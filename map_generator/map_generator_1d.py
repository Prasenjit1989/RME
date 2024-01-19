import random
import numpy as np


class MapGenerator1D:
    def __init__(self, n_tx = 1, r_x_min = 1, r_x_max = 100, alpha = 30):
        self.alpha = alpha
        self.n_tx = n_tx
        #self.tx_locations = tx_locations
        self.r_x_min = r_x_min
        self.r_x_max = r_x_max
    
    def generate_tx_locations(self):
        return np.array([[random.uniform(self.r_x_min, self.r_x_max), random.uniform(self.r_x_min, self.r_x_max)]
            for _ in range(self.n_tx)])
        
    def generate_map(self, num_points, y_coord = 0):
        m_tx_locs = self.generate_tx_locations()
        v_x = np.linspace(self.r_x_min, self.r_x_max, num_points)
        m_x_coord_distances = np.tile(v_x, (self.n_tx, 1)) - np.tile(m_tx_locs[:, 0].reshape(-1, 1), (1, v_x.shape[0]))
        m_betas = np.tile(m_tx_locs[:, 1].reshape(-1, 1), (1, v_x.shape[0])) - y_coord
        v_received_pow = np.sum(self.alpha/ (m_x_coord_distances**2 + m_betas**2), axis = 0)        
        
        return m_tx_locs, v_x, v_received_pow
        

    
    def generate_noise(self, noise_level = 0.1, num_points = 100):
        return noise_level * np.random.uniform(0, 0.1, size = num_points)

    
    def generate_measurement_locations(self, num_samples = 20, num_points = 100):
        return np.array(sorted(np.random.choice(num_points, size = num_samples, replace = False)))
        # return np.array(sorted(np.random.randint(0, num_points, num_samples)))
    
    def generate_sampling_mask(self, num_samples = 20, num_points =100):
            v_measurement_locs = self.generate_measurement_locations(num_samples, num_points)
            v_sampling_mask = np.zeros(num_points)
            v_sampling_mask[v_measurement_locs] = 1 
            
            return v_sampling_mask, v_measurement_locs
    
    def generate_measurement_data(self, v_x, v_true_map, v_noise, v_sampling_mask):
        return  np.multiply(v_true_map + v_noise, v_sampling_mask)[np.where(v_sampling_mask == 1)] 

