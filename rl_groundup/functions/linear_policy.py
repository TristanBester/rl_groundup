# Created by Tristan Bester.
import numpy as np

class LinearPolicy(object):
    '''A parametric function used to approximate the state-action value
    function for a specific policy. The approximate value for a given state-
    action pair is equal to the dot product of the state-action pairs feature
    vector and the parameter vector of the function.
    '''
    def __init__(self, n_weights, action_vec_dim, n_actions):
        self.weights = np.zeros(n_weights + action_vec_dim)
        self.n_actions = n_actions


    def evaluate(self, feature_vector):
        return np.dot(self.weights, feature_vector)


    def greedy_action(self, feature_vectors):
        action_values = [self.evaluate(x) for x in feature_vectors]
        return np.argmax(action_values)
