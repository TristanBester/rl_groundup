# Created by Tristan Bester.
import sys
import numpy as np
sys.path.append('../')
from envs import GridWorld
from utils import print_grid_world_actions

'''
Value iteration has been used to the grid world problem defined on page 61
of "Reinforcement Learning: An Introduction."
Algorithm available on page 67.

Book reference:
Sutton, R. and Barto, A., 2014. Reinforcement Learning:
An Introduction. 1st ed. London: The MIT Press.
'''


def value_iteration(env, gamma, epsilon):
    '''Value iteration.'''
    n_states = env.observation_space_size
    n_actions =  env.action_space_size
    V = np.zeros(n_states)
    delta = np.inf

    while delta > epsilon:
        V_last = V.copy()
        for s in range(n_states):
            v = []
            for a in range(n_actions):
                proba, n_state, reward, done = env.P[s,a]
                v.append(proba * (reward + gamma * V[n_state]))
            V[s] = np.max(v)
        delta = np.max(abs(V - V_last))
    return V


def get_deterministic_policy(env, V):
    '''Return a policy that is greedy w.r.t. the given value function.'''
    n_states = env.observation_space_size
    n_actions = env.action_space_size
    policy = np.zeros((n_states, n_actions), dtype='float32')

    for s in range(1,n_states-1):
        possible_states = np.full((4,1), -np.inf)
        for a in range(n_actions):
            n_state = env.P[s,a][1]
            if n_state != s:
                possible_states[a] = V[n_state]
        best_action = np.argmax(possible_states)
        policy[s][best_action] = 1.0
    return policy


if __name__ == '__main__':
    env = GridWorld()
    epsilon = 1e-5
    gamma = 1
    V = value_iteration(env, gamma, epsilon)
    policy = get_deterministic_policy(env, V)
    print_grid_world_actions(policy)
    print(f'Optimal value function:\n{V.reshape(4,4)}')
