# Created by Tristan Bester.
import sys
import numpy as np
sys.path.append('../')
from envs import GridWorld
from utils import print_grid_world_actions
from policy_evaluation import policy_evaluation

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
    policy = np.zeros((env.observation_space_size, env.action_space_size),\
                       dtype='float32')
    policy_stable = True

    for s in range(1,env.observation_space_size-1):
        # Initialize action values.
        possible_states = np.full((4,1), -np.inf)
        for a in range(env.action_space_size):
            # Get new state after taking action a in state s.
            n_state = env.P[s,a][1]
            # If action a is valid store action value.
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
    policy_ls = [np.argmax(ls) for ls in policy]
    print_grid_world_actions(policy_ls, (4,4), [0, 15])
    print(f'Optimal value function:\n{V_star}')
