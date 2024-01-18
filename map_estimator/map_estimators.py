from sklearn.neighbors import KNeighborsRegressor

class MapEstimator:
    # def __init__(self, v_measurement_locations, v_measurements, v_grid_x_points):
    #     self.v_measurement_locations = v_measurement_locations
    #     self.v_measurements = v_measurements
    #     self.v_grid_x_points = v_grid_x_points
        
    def estimate(self, v_measurement_locations, v_measurements, v_grid_x_points):
        # write common funcitons
        return self._estimate(v_measurement_locations, v_measurements, v_grid_x_points)
    
    def _estimate(self, v_measurement_locations, v_measurements, v_grid_x_points):
        raise NotImplementedError


class KnnEstimator(MapEstimator):

    def __init__(self, num_neighbors=5):
        self.num_neighbors = num_neighbors
        self.knn_reg = KNeighborsRegressor(n_neighbors=self.num_neighbors, weights='distance')

    def _estimate(self, v_measurement_locations, v_measurements, v_grid_x_points):        
        # X_train, X_test = v_measurement_locations, v_grid_x_points        
        self.knn_reg.fit(v_measurement_locations, v_measurements)
        return self.knn_reg.predict(v_grid_x_points)



# class LinearInterpolationEstimator(MapEstimator):
#     def 