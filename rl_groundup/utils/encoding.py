# Created by Tristan Bester.
import numpy as np

def encode_state(curr_state, n_states):
    '''Return a vector with a component for each possible state. All components
    values are set to zero except for the component corresponding to the agents
    current state, which has a value of one.'''
    vec = np.zeros(n_states)
    vec[curr_state] = 1
    return vec


def encode_sa_pair(state, action, n_states, n_actions):
    '''Return a vector with a component for each possible state-action pair.
    All components values are set to zero except for the component corresponding
    to the current state-action pair.'''
    vec = np.zeros((n_states, n_actions))
    vec[state][action] = 1
    return vec.flatten()
