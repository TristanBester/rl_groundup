# Created by Tristan Bester.
import sys
import numpy as np
sys.path.append('../')
from itertools import product
from envs import WindyGridWorld
from utils import print_episode, eps_greedy_policy, test_policy, \
                  create_greedy_policy

'''
n-step Sarsa used to estimate the optimal policy for
the windy gridworld environment defined on page 106 of
"Reinforcement Learning: An Introduction."
Algorithm available on page 120.

Book reference:
Sutton, R. and Barto, A., 2014. Reinforcement Learning:
An Introduction. 1st ed. London: The MIT Press.
'''


def n_step_sarsa(env, n, alpha, gamma, epsilon, n_episodes):
    # Initialize state-action value function.
    sa_pairs = product(range(env.observation_space_size), \
                       range(env.action_space_size))
    Q = dict.fromkeys(sa_pairs, 0.0)
    states = np.zeros(n)
    actions =np.zeros(n)
    rewards = np.zeros(n)

    for episode in range(n_episodes):
        done = False
        obs = env.reset()
        t = 0
        T = np.inf
        states[t] = obs
        a = eps_greedy_policy(Q, obs, epsilon, env.action_space_size)
        actions[t] = a

        step_count = 0
        while not done or tau != T-1:
            step_count += 1
            if t < T:
                obs_prime, reward, done = env.step(a)
                rewards[(t+1)%n] = reward
                states[(t+1)%n] = obs_prime
                if done:
                    T = t+1
                else:
                    a = eps_greedy_policy(Q, obs_prime, epsilon,\
                                          env.action_space_size)
                    actions[(t+1)%n] = a
            tau = t - n + 1
            if tau > -1:
                G = np.sum([gamma ** (i-tau-1) * rewards[i%n] for i in range(tau + 1, min(tau+n, T))])
                if tau + n < T:
                    state = states[(tau+n)%n]
                    action = actions[(tau+n)%n]
                    G += gamma ** n * Q[state, action]
                s = states[tau%n]
                a = actions[tau%n]
                # Update state-action value estimate.
                Q[s,a] += alpha * (G - Q[s,a])
            t += 1
        if episode % 1 == 0:
            print_episode(episode, n_episodes)
    print_episode(n_episodes, n_episodes)
    return Q


if __name__ == '__main__':
    n = 5
    alpha = 0.4
    gamma = 0.7
    epsilon = 0.1
    n_episodes = 5000
    n_tests = 30
    env = WindyGridWorld()
    Q = n_step_sarsa(env, n, alpha, gamma, epsilon, n_episodes)
    policy = create_greedy_policy(Q, env.observation_space_size, env.action_space_size)
    test_policy(env, policy, n_tests)
