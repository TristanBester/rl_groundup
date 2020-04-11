# Created by Tristan Bester.
import sys
import numpy as np
sys.path.append('../')
from envs import GridWorld
from utils import create_value_func_plot

'''
Policy evalation has been used to calculate the value function for a random
policy in the grid world environment defined on page 61
of "Reinforcement Learning: An Introduction."
Algorithm available on page 61.

Book reference:
Sutton, R. and Barto, A., 2014. Reinforcement Learning:
An Introduction. 1st ed. London: The MIT Press.
'''


def policy_evaluation(env, policy, epsilon, gamma):
    '''Policy evaluation algorithm.'''
    n_states = env.observation_space_size
    n_actions = env.action_space_size
    V = np.zeros(n_states)
    delta = np.inf

    while delta > epsilon:
        V_last = V.copy()
        for s in range(n_states):
            val = 0
            for a in range(n_actions):
                proba, n_state, reward, done = env.P[s,a]
                val += policy[s,a] * proba * (reward + gamma * V_last[n_state])
            V[s] = val
        delta = np.max(abs(V - V_last))

    return V


if __name__ == '__main__':
    env = GridWorld()
    # All actions have eqaul probabilities of being taken.
    policy = np.full((env.observation_space_size, env.action_space_size), 1/env.action_space_size)
    epsilon = 1e-5
    gamma = 1

    V = policy_evaluation(env, policy, epsilon, gamma)
    title = 'Grid world value function:'
    create_value_func_plot(V, (4,4), title)
