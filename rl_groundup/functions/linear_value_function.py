import numpy as np

class LinearValueFunction(object):
    def __init__(self, n_weights):
        self.weights = np.zeros(n_weights)
        
    
    def evaluate(self, feature_vector):
        return np.dot(self.weights, feature_vector)