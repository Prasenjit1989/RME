import random
import numpy as np

class MapGenerator1D:
    def __init__(self, n_tx, tx_locations, r_x_min = 1, r_x_max = 100, alpha = 30):
        self.alpha = alpha
        self.n_tx = n_tx
        #self.tx_locations = tx_locations
        self.r_x_min = r_x_min
        self.r_x_max = r_x_max
    
    def generate_tx_locations(self):
        return np.array([[random.uniform(self.r_x_min, self.r_x_max), random.uniform(self.r_x_min, self.r_x_max)]
            for _ in range(self.n_tx)])
        
    def generate_map(self, num_points, num_maps = 1, y_coord = 0):
        list_m_tx_locs, list_v_x, list_v_received_pow, list_y_coord = [], [], [], []
        for _ in num_maps:
            m_tx_locs = self.generate_tx_locations()
            v_x = np.linspace(self.r_x_min, self.r_x_max, num_points)
            m_x_coord_distances = np.tile(v_x, (self.n_tx, 1)) - np.tile(m_tx_locs[:, 0].reshape(-1, 1), (1, v_x.shape[0]))
            m_betas = np.tile(m_tx_locs[:, 1].reshape(-1, 1), (1, v_x.shape[0])) - y_coord
            v_received_power = np.sum(self.alpha/ (m_x_coord_distances**2 + m_betas**2), axis = 0)
            
            list_m_tx_locs.append(m_tx_locs) # tx locations and received power
            list_v_x.append(v_x)
            list_v_received_pow.append(v_received_power)
            list_y_coord.append(y_coord)

        return {'tx_locs': list_m_tx_locs, 'v_x': list_v_x, 'v_received_power': list_v_received_pow, 'y_coord': y_coord}
    
    def generate_noise(self, num_points, num_maps = 1):
        return np.random.uniform(0, 0.1, size = (num_maps, num_points))      

    def generate_sampling_mask(self, num_samples, num_points, num_maps = 1):
        m_sampling_mask = []
        for _ in num_maps:
            v_mask = [1]*num_samples [0]*(num_points - num_samples)
            random.shuffle(v_mask)
            m_sampling_mask.append(v_mask)
        return np.array(m_sampling_mask)


