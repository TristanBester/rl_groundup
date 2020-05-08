import numpy as np 

class LinearPolicy(object):
    def __init__(self, n_weights, action_vec_dim, n_actions):
        self.weights = np.zeros(n_weights + action_vec_dim)
        self.n_actions = n_actions
        
    def evaluate(self, feature_vector):
        return np.dot(self.weights, feature_vector)
    
    def greedy_action(self, feature_vectors):
        action_values = [self.evaluate(x) for x in feature_vectors]
        return np.argmax(action_values)
        
    