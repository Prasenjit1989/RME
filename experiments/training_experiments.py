import numpy as np
import matplotlib.pyplot as plt
import random

import gsim
from gsim.gfigure import GFigure


class ExperimentSet(gsim.AbstractExperimentSet):

    def experiment_1001(l_args):
        """
        TODO:

        - Fine grid for the true map. 

        - Measurements are taken on a small number of locations (e.g. 20)

        - Plot measurements with an x marker. 

        - Plot the true map and the estimated map on the fine grid. 

        - Code each estimator as a subclass of a base class MapEstimator. The
          parameters such as degree or number of neighbors should be passed to
          the constructor. A method `estimate` should be implemented for each
          subclass and the same parameters should be passed to it.

        - After doing this, create an experiment function where the DNN
          estimator is trained. In the current experiment, the DNN estimator
          just loads the weights upon instantiation (pass file name of the
          weights to the constructor).

          
        -------------------------------------------
        Pseudocode for this experiment

        map_generator = MapGenerator1D(args)

        v_true_map = map_generator.generate_map()

        v_meas, v_meas_locs = get_measurements(v_true_map)

        l_estimators = [PolynomialEstimator(args), KNN(args)]

        l_estimates = []
        for estimator in l_estimators:
            v_estimated_map = estimator.estimate(v_meas, v_meas_locs)
            l_estimates.append(v_estimated_map)

        plot(v_true_map, v_meas, v_meas_locs, l_estimates)

        
        




        """

        def Polynomial_RME_estimation(measurements,
                                      measurement_locs,
                                      grid_x_test,
                                      degree=3):

            # Construct the polynomial features
            X_poly = np.vander(measurement_locs, degree + 1, increasing=True)

            # Perform linear regression
            estimated_coeff = np.linalg.lstsq(X_poly, measurements,
                                              rcond=None)[0]

            # construct the polynomial features for the grid
            grid_x_poly = np.vander(grid_x_test, degree + 1, increasing=True)

            # Predicted power
            predicted_power = np.dot(grid_x_poly, estimated_coeff)

            return predicted_power

        def linear_interpolation_RME_estimation(measurements, measurement_locs,
                                                grid_x_test):

            predicted_power = np.zeros(grid_x_test.shape)
            for i in range(len(grid_x_test)):
                k = np.searchsorted(measurement_locs,
                                    grid_x_test[i],
                                    side='right')
                if k == 0:
                    predicted_power[i] = measurements[0]
                elif k == len(measurements):
                    predicted_power[i] = measurements[-1]
                else:
                    x_left, x_right = measurement_locs[k -
                                                       1], measurement_locs[k]
                    y_left, y_right = measurements[k - 1], measurements[k]
                    predicted_power[i] = y_left + (grid_x_test[i] - x_left) * (
                        y_right - y_left) / (x_right - x_left)

            return predicted_power

        def KNN_RME_estimation(measurements,
                               measurement_locs,
                               grid_x_test,
                               K_neighbors=5):
            from sklearn.neighbors import KNeighborsRegressor
            X_train, X_test = measurement_locs.reshape(-1,
                                                       1), grid_x_test.reshape(
                                                           -1, 1)
            knn_reg = KNeighborsRegressor(n_neighbors=K_neighbors,
                                          weights='distance')
            knn_reg.fit(X_train, measurements)

            predicted_power = knn_reg.predict(X_test)
            return predicted_power

        def DNN_RME_estimation(measurements, measurement_locs, grid_x_test):
            import tensorflow as tf
            X_train, X_test = measurement_locs.reshape(-1,
                                                       1), grid_x_test.reshape(
                                                           -1, 1)
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(32,
                                      activation='relu',
                                      input_shape=(X_train.shape[1], )),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(1)
            ])

            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, measurements, epochs=50, batch_size=16)
            predicted_power = model.predict(X_test)

            return predicted_power

        alpha_gain = 20  # transmitted power * (lambda/(4*pi))**2
        r_x_min, r_x_max = 1, 200
        num_samples = 80
        num_transmitters = 1
        random.seed(2)

        # locations of the transmitters
        # TODO: generate 2D tx. locations. Compute the betas from those locations.
        transmitters_locs, beta_s = np.array([
            round(random.uniform(r_x_min + 10, r_x_max - 10), 3)
            for _ in range(num_transmitters)
        ]).reshape(1, -1), np.array(random.uniform(0, 1))

        # Generating measurements at known transmitter locations
        v_measurement_locs = np.array(
            sorted(
                np.round(np.random.uniform(r_x_min, r_x_max, num_samples), 2)))

        # TODO: the true map must be created using a function. The samples are obtained
        # by evaluating the true map and then adding noise, which can be done in another
        # function.

        # Creating measurement matrix for holding measurement data from different transmitters
        m_measurements = np.zeros((num_samples, num_transmitters))
        distances = np.abs(
            np.tile(v_measurement_locs[:, np.newaxis], (1, num_transmitters)) -
            np.tile(transmitters_locs, (num_samples, 1)))

        # Generating measurement data from true coefficients and distance values
        # TODO: put the measurements in a vector (1D)
        noise_level = 0.01
        m_measurements = alpha_gain / (distances +
                                       beta_s**2) + np.random.randn(
                                           num_samples, 1) * noise_level
        measurements_log = 10 * np.log10(m_measurements)

        # Create a grid for prediction
        grid_size = 300
        grid_x_test = np.linspace(r_x_min, r_x_max, grid_size)

        # Estimation using different methods
        degree = 5
        estimated_power_poly = Polynomial_RME_estimation(measurements_log,
                                                         v_measurement_locs,
                                                         grid_x_test,
                                                         degree=degree)
        estimated_power_interp = linear_interpolation_RME_estimation(
            measurements_log, v_measurement_locs, grid_x_test)
        estimated_power_knn = KNN_RME_estimation(measurements_log,
                                                 v_measurement_locs,
                                                 grid_x_test,
                                                 K_neighbors=5)
        estimated_power_dnn = DNN_RME_estimation(measurements_log,
                                                 v_measurement_locs,
                                                 grid_x_test)

        plt.figure(figsize=(8, 6))
        plt.plot(v_measurement_locs,
                 measurements_log,
                 linestyle="solid",
                 color="g",
                 label='Measurements')
        plt.plot(grid_x_test,
                 estimated_power_poly,
                 linestyle="dashdot",
                 color="r",
                 label='Estimated Map (Polynomial)')
        plt.plot(grid_x_test,
                 estimated_power_interp,
                 linestyle="dashdot",
                 color="m",
                 label='Estimated Map (Interpolation)')
        plt.plot(grid_x_test,
                 estimated_power_knn,
                 linestyle="dashdot",
                 color="y",
                 label='Estimated Map (KNN)')
        plt.plot(grid_x_test,
                 estimated_power_dnn,
                 linestyle="dashdot",
                 color="c",
                 label='Estimated Map (DNN)')

        plt.xlabel("Locations")
        plt.ylabel("Power (dB)")
        plt.title("True vs Estimated Power")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.show()

    def experiment_1002(l_args):
        """ Experiment to train a DNN estimator. 
        
        Pseudocode:

        1. GENEREATE THE DATA SET

        num_maps = 1000
        map_generator = MapGenerator1D(args)
        l_dataset = []

        for ind_map in range(num_maps):
        
            v_true_map = map_generator.generate_map()
            v_meas, v_meas_locs = get_measurements(v_true_map)

            m_input = [v_meas, v_meas_locs]
            v_target = v_true_map

            l_dataset.append((m_input, v_target)))

            
        2. TRAIN THE DNN ESTIMATOR

        estimator = DNN_RME_estimation(args)

        training_history = estimator.train(l_dataset)

        plot(training_history) # training and validation loss vs. epoch index
        
    
        """

    ######### TEST EXPERIMENTS #########
    def experiment_10002(l_args):
        """The full potential of GSim is exploited by returning GFigures
        rather than directly plotting them. GFigures are stored and
        can be plotted and edited afterwards without having to run
        again the experiment.

        GFigure offers a neater interface than matplotlib, whose goal
        was to resemble MATLAB's interface. 

        See gsim.gfigure.example_figures for examples on how to use
        GFigure.

        """
        print("This experiment plots a figure.")

        v_x = np.linspace(0, 10, 20)
        v_y1 = v_x**2 - v_x + 3

        # Example with a single curve, single subplot
        G = GFigure(xaxis=v_x,
                    yaxis=v_y1,
                    xlabel="x",
                    ylabel="f(x)",
                    title="Parabola",
                    legend="P1")

        return G

    def experiment_10003(l_args):
        """ 
        In some occasions, it may be useful to access a GFigure created by a
        previously-run experiment; e.g. to combine multiple figures. 
        """

        l_G = ExperimentSet.load_GFigures(1002)
        G = GFigure()
        # Same subplot twice
        G.l_subplots = l_G[0].l_subplots + l_G[0].l_subplots
        return G
