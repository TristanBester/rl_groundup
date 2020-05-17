# Created by Tristan Bester.
import sys
import numpy as np
sys.path.append('../')
from envs import GridWorld
from functions import LinearPolicy
from utils import print_episode, encode_sa_pair, test_linear_policy, \
                  eps_greedy_policy_bin_features

'''
True Online Sarsa lambda with binary features and linear function approximation
used to estimate the optimal policy for the Gridworld environment defined of
page 48 of "Reinforcement Learning: An Introduction."
The algorithm can be found on page 252 of the same text.

Book reference:
Sutton, R. and Barto, A., 2014. Reinforcement Learning:
An Introduction. 1st ed. London: The MIT Press.
'''


def online_sarsa_lambda(env, lamda, alpha, gamma, epsilon, n_episodes):
    # Initialize state-action value function.
    q = LinearPolicy(env.observation_space_size * env.action_space_size, 0,
                     env.action_space_size)

    for episode in range(n_episodes):
        done = False
        obs = env.reset()
        a = eps_greedy_policy_bin_features(q, obs, epsilon, env.observation_space_size, \
                                           env.action_space_size)
        x = encode_sa_pair(obs, a, env.observation_space_size, env.action_space_size)
        z = np.zeros(env.observation_space_size * env.action_space_size)
        Q_old = 0

        while not done:
            obs_prime, reward, done = env.step(a)
            a_prime = eps_greedy_policy_bin_features(q, obs_prime, epsilon, \
                      env.observation_space_size, env.action_space_size)
            x_prime = encode_sa_pair(obs_prime, a_prime, env.observation_space_size, \
                                     env.action_space_size)
            Q = q.evaluate(x)
            Q_prime = q.evaluate(x_prime)
            delta = reward + gamma * Q_prime - Q
            # Update eligibility traces.
            z = gamma * lamda * z + (1 - alpha * gamma * lamda * np.dot(z, x)) * x
            # Update weights.
            q.weights += alpha * (delta  + Q - Q_old) * z - alpha * (Q - Q_old) * x
            Q_old = Q
            x = x_prime
            a = a_prime
        if episode % 100 == 0:
            print_episode(episode, n_episodes)
    print_episode(n_episodes, n_episodes)
    return q


if __name__ == '__main__':
    gamma = 1
    alpha = 0.001
    lamda = 0.5
    epsilon = 0.1
    n_episodes = 10000
    env = GridWorld()
    q = online_sarsa_lambda(env, lamda, alpha, gamma, epsilon, n_episodes)
    test_linear_policy(env, q, 5)
