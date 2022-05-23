# k_nearest_neighbors.py: Machine learning implementation of a K-Nearest Neighbors classifier from scratch.
#
# Submitted by: [Ritwik Budhiraja] -- [rbudhira]
#
# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff

import numpy as np
from utils import euclidean_distance, manhattan_distance


class KNearestNeighbors:
    """
    A class representing the machine learning implementation of a K-Nearest Neighbors classifier from scratch.

    Attributes:
        n_neighbors
            An integer representing the number of neighbors a sample is compared with when predicting target class
            values.

        weights
            A string representing the weight function used when predicting target class values. The possible options are
            {'uniform', 'distance'}.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model and
            predicting target class values.

        _y
            A numpy array of shape (n_samples,) representing the true class values for each sample in the input data
            used when fitting the model and predicting target class values.

        _distance
            An attribute representing which distance metric is used to calculate distances between samples. This is set
            when creating the object to either the euclidean_distance or manhattan_distance functions defined in
            utils.py based on what argument is passed into the metric parameter of the class.

    Methods:
        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """

    def __init__(self, n_neighbors = 5, weights = 'uniform', metric = 'l2'):
        # Check if the provided arguments are valid
        if weights not in ['uniform', 'distance'] or metric not in ['l1', 'l2'] or not isinstance(n_neighbors, int):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the KNearestNeighbors model object
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._X = None
        self._y = None
        self._distance = euclidean_distance if metric == 'l2' else manhattan_distance

    def fit(self, X, y):
        """
        Fits the model to the provided data matrix X and targets y.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """
        self._X = X
        self._y = y
    
    # Reference: https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
    def predict(self, X):
        """
        Predicts class target values for the given test data matrix X using the fitted classifier model.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the test data.

        Returns:
            A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
        """
        smoothing=0.1    # smoothing parameter
        pred = []
        for x in X:
            class_count = [0] * len(set(self._y))
            neighbour_dist_class = []

            for i in range(len(self._X)):
                dist = self._distance(self._X[i], x)    # calculating distance for x_i with other data points
                neighbour_dist_class.append((dist, self._y[i]))

            neighbour_dist_class.sort(key=lambda ele:ele[0])    # sorting for majority class
            nearest_neighbours = neighbour_dist_class[:self.n_neighbors]    # getting top k nearest neighbours

            if self.weights == 'uniform':
                for neighbour in nearest_neighbours:
                    class_count[neighbour[1]] += 1    # weighting nearest neighbour class counts uniformly
            else:
                for neighbour in nearest_neighbours:
                    class_count[neighbour[1]] += 1/(neighbour[0]+smoothing)    # weighting nearest neighbours by reciprocal of distance

            pred.append(class_count.index(max(class_count)))    # appending majority class as prediction

        return np.array(pred)
        
