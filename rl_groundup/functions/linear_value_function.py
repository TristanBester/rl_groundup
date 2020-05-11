# Created by Tristan Bester.
import numpy as np

class LinearValueFunction(object):
    '''A parametric function used to approximate the state-value function
    under a specific policy. The approximate value for a given state
    is equal to the dot product of the states feature vector and the
    parameter vector of the function.
    '''
    def __init__(self, n_weights):
        self.weights = np.zeros(n_weights)


    def evaluate(self, feature_vector):
        return np.dot(self.weights, feature_vector)
