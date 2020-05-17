# Created by Tristan Bester.
import sys
import numpy as np
sys.path.append('../')
from envs import GridWorld
from functions import LinearPolicy
from utils import print_episode, encode_sa_pair, test_linear_policy, \
                  eps_greedy_policy_bin_features

'''
Sarsa lambda with binary features and linear function approximation used to
estimate the optimal policy for the Gridworld environment defined of page 48
of "Reinforcement Learning: An Introduction."
The algorithm can be found on page 250 of the same text.

Book reference:
Sutton, R. and Barto, A., 2014. Reinforcement Learning:
An Introduction. 1st ed. London: The MIT Press.
'''


def sarsa_lambda(env, lamda, alpha, gamma, epsilon, n_episodes):
    # Initialize state-action value function.
    q = LinearPolicy(env.observation_space_size * env.action_space_size, 0, \
                     env.action_space_size)

    for episode in range(n_episodes):
        done = False
        obs = env.reset()
        action = eps_greedy_policy_bin_features(q, obs, epsilon, \
                 env.observation_space_size, env.action_space_size)
        z = np.zeros(env.observation_space_size * env.action_space_size)

        while not done:
            obs_prime, reward, done = env.step(action)
            delta = reward
            sa_vec = encode_sa_pair(obs, action, env.observation_space_size, \
                                    env.action_space_size)
            idx_active = np.argwhere(sa_vec == 1)
            delta -= np.sum(q.weights[idx_active])
            # Accumulating traces.
            z[idx_active] += 1

            if done:
                # Update weights.
                q.weights += alpha * delta * z
            else:
                action_prime = eps_greedy_policy_bin_features(q, obs_prime, epsilon, \
                               env.observation_space_size, env.action_space_size)
                sa_prime_vec = encode_sa_pair(obs_prime, action_prime, \
                               env.observation_space_size, env.action_space_size)
                idx_active = np.argwhere(sa_prime_vec == 1)
                delta += gamma * np.sum(q.weights[idx_active])
                # Update weights.
                q.weights += alpha * delta * z
                # Update accumulating traces.
                z = gamma * lamda * z
                obs = obs_prime
                action = action_prime
        if episode % 100 == 0:
            print_episode(episode, n_episodes)
    print_episode(n_episodes, n_episodes)
    return q


if __name__ == '__main__':
    gamma = 1
    lamda = 0.5
    alpha = 0.001
    epsilon = 0.1
    n_episodes = 10000
    env = GridWorld()
    q = sarsa_lambda(env, lamda, alpha, gamma, epsilon, n_episodes)
    test_linear_policy(env, q, 5)
