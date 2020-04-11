# Created by Tristan Bester.
import sys
import numpy as np
sys.path.append('../')
from envs import GridWorld
from policy_evaluation import policy_evaluation
from utils import print_grid_world_actions

'''
Policy iteration has been used to the grid world problem defined on page 61
of "Reinforcement Learning: An Introduction."
Algorithm available on page 65.

Book reference:
Sutton, R. and Barto, A., 2014. Reinforcement Learning:
An Introduction. 1st ed. London: The MIT Press.
'''


def improve_policy(env, V, old_policy):
    '''Return a policy that is greedy w.r.t. the given value function.'''
    n_states = env.observation_space_size
    n_actions = env.action_space_size
    policy = np.zeros((n_states, n_actions), dtype='float32')
    policy_stable = True

    for s in range(1,n_states-1):
        possible_states = np.full((4,1), -np.inf)
        for a in range(n_actions):
            n_state = env.P[s,a][1]
            if n_state != s:
                possible_states[a] = V[n_state]
        best_action = np.argmax(possible_states[:])
        policy[s][best_action] = 1.0
        if not np.all(old_policy[s] == policy[s]):
            policy_stable = False

    return policy, policy_stable


def policy_iteration(env, policy, epsilon, gamma):
    '''Policy iteration.'''
    n_states = env.observation_space_size
    n_actions = env.action_space_size
    policy_stable = False

    while not policy_stable:
        V = policy_evaluation(env, policy, epsilon, gamma)
        policy, policy_stable = improve_policy(env,V, policy)

    return policy


if __name__ == '__main__':
    gamma = 1.0
    epsilon = 1e-5
    env = GridWorld()
    n_states = env.observation_space_size
    n_actions = env.action_space_size
    policy = np.full((n_states, n_actions), 1/n_actions)
    policy = policy_iteration(env, policy, epsilon, gamma)
    V_star = policy_evaluation(env, policy, epsilon,gamma).reshape(4,4)
    print_grid_world_actions(policy)
    print(f'Optimal value function:\n{V_star}')
