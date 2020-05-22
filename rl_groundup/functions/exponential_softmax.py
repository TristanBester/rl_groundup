import numpy as np

class ExponentialSoftmax(object):
    '''Linear numerical preferences.'''
    def __init__(self, feature_vector_shape):
        self.weights = np.full(feature_vector_shape, -10.0)

    def numerical_preference(self, feature_vector):
        return np.dot(self.weights, feature_vector)

    def get_preferences(self, all_feature_vectors):
        preferences = [self.numerical_preference(v) for v in all_feature_vectors]
        min = np.min(preferences)
        max = np.max(preferences)
        if min != max:
            preferences = (preferences - min)/(max - min)
        return preferences

    def sample_action(self, all_feature_vectors):
        preferences = self.get_preferences(all_feature_vectors)

        #print(preferences)
        vals = [np.exp(h) for h in preferences]
        #print(vals)

        sum = np.sum(vals)
        #print(sum)
        #print()
        if sum != 0:
            probas = [v/sum for v in vals]
            return np.random.choice(range(len(preferences)), p=probas)
        else:
            return np.random.choice(range(len(preferences)))



    def greedy_action(self, all_feature_vectors):
        preferences = self.get_preferences(all_feature_vectors)
        vals = [np.exp(h) for h in preferences]
        summ = np.sum(vals)
        probas = [v/summ for v in vals]
        return np.argmax(probas)

    def eligibility_vector(self, action, all_feature_vectors):
        preferences = self.get_preferences(all_feature_vectors)
        probas = [np.exp(h) for h in preferences]
        scaled_vectors = []

        for i in range(len(all_feature_vectors)):
            scaled_vectors.append(probas[i] * all_feature_vectors[i])

        summ = np.sum(scaled_vectors, axis=0)
        '''print(all_feature_vectors[action].reshape(16,4))
        print(summ.reshape(16,4))'''
        mod_vec = (all_feature_vectors[action] - summ)
        #print()
        #input()





        '''vals = [np.exp(preferences[i])*all_feature_vectors[i] for i in \
                range(len(all_feature_vectors))]
        vals = np.sum(vals, axis=0)'''
        #return all_feature_vectors[action] - vals
        return mod_vec
