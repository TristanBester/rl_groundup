# Created by Tristan Bester.
import sys
import numpy as np
sys.path.append('../')
from itertools import product
from envs import WindyGridWorld
from td_zero_prediction import td_pred
from utils import print_episode, create_value_func_plot, eps_greedy_policy, \
                  create_greedy_policy, print_grid_world_actions, test_policy

'''
Sarsa (on-policy TD control) used to estimate the optimal policy for
the windy gridworld environment defined on page 106 of
"Reinforcement Learning: An Introduction."
Algorithm available on page 106.

Book reference:
Sutton, R. and Barto, A., 2014. Reinforcement Learning:
An Introduction. 1st ed. London: The MIT Press.
'''


def sarsa(env, gamma, alpha, epsilon, n_episodes):
    # Create iterators.
    sa_pairs = product(range(70), range(4))
    # Initialize state-action value function.
    Q = dict.fromkeys(sa_pairs, 0.0)

    epsilon_start = epsilon
    decay = lambda x: x - (10/n_episodes)*epsilon_start if \
            x - (10/n_episodes)*epsilon_start > 1e-4 else 1e-4

    for episode in range(n_episodes):
        done = False
        obs = env.reset()
        action = eps_greedy_policy(Q, obs, epsilon, env.action_space_size)

        while not done:
            obs_prime, reward, done = env.step(action)
            action_prime = eps_greedy_policy(Q, obs_prime, epsilon, \
                                             env.action_space_size)
            # Update state-action value estimate.
            Q[obs,action] += alpha * (reward + gamma * \
                             (Q[obs_prime, action_prime]) - Q[obs, action])
            obs = obs_prime
            action = action_prime

        # Decay epsilon.
        epsilon = decay(epsilon)
        if episode % 100 == 0:
            print_episode(episode, n_episodes)
    print_episode(n_episodes, n_episodes)
    return Q


if __name__ == '__main__':
    alpha = 0.4
    gamma = 0.7
    epsilon = 1.0
    n_episodes = 15000
    n_tests = 10
    env = WindyGridWorld()

    print('Beginning control...\n')
    Q = sarsa(env, gamma, alpha,epsilon, n_episodes)
    policy = create_greedy_policy(Q, env.observation_space_size, \
                                  env.action_space_size)
    test_policy(env, policy, n_tests)
