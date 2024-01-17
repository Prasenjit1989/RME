class MapEstimator:
    def __init__(self, v_measurement_locations, v_measurements, v_grid_x_points):
        self.v_measurement_locations = v_measurement_locations
        self.v_measurements = v_measurements
        self.v_grid_x_points = v_grid_x_points
        
    def estimate(self, v_measurement_locations, v_measurements, v_grid_x_points):
        # write common funcitons
        estimated_maps = self._estimate(v_measurement_locations, v_measurements, v_grid_x_points)
    
    def _estimate(self, v_measurement_locations, v_measurements, v_grid_x_points):
        raise NotImplementedError


class KnnEstimator(MapEstimator):
    def __init__(self, num_neighbors=5):
        self.num_neighbors = num_neighbors

    def _estimate(self):
        from sklearn.neighbors import KNeighborsRegressor
        X_train, X_test = self.v_measurement_locations, self.v_grid_x_points
        knn_reg = KNeighborsRegressor(n_neighbors=self.num_neighbors, weights='distance')
        knn_reg.fit(X_train, self.v_measurements)
        return knn_reg.predict(self.v_grid_x_points)