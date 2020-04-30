# Created by Tristan Bester.
import sys
import numpy as np
sys.path.append('../')
from envs import GridWorld
from itertools import product
from utils import print_episode, eps_greedy_policy, test_policy

'''
n-step Tree Backup used to estimate the optimal policy for
the gridworld environment defined on page 48 of
"Reinforcement Learning: An Introduction."
Algorithm available on page 125.

Book reference:
Sutton, R. and Barto, A., 2014. Reinforcement Learning:
An Introduction. 1st ed. London: The MIT Press.
'''


def policy_proba(policy, s, a, epsilon):
    '''Return the probability of the given epsilon-greedy policy
    taking the specified action in the specified state.'''
    if policy[s] == a:
        return (epsilon/4) + (1-epsilon)
    else:
        return epsilon/4


def n_step_tree_backup(env, n, alpha, gamma, epsilon, n_episodes):
    # Initialize policy and state-action value function.
    sa_pairs = product(range(env.observation_space_size),\
                       range(env.action_space_size))
    Q = dict.fromkeys(sa_pairs, 0.0)
    policy = dict.fromkeys(range(env.observation_space_size), 0)
    states = np.zeros(n)
    actions = np.zeros(n)
    Qs = np.zeros(n)
    deltas = np.zeros(n)
    pis = np.zeros(n)

    decay = lambda x: x-2/n_episodes if x-2/n_episodes > 0.1 else 0.1

    for episode in range(n_episodes):
        done = False
        obs = env.reset()
        action = eps_greedy_policy(Q, obs, epsilon, env.action_space_size)
        states[0] = obs
        actions[0] = action
        Qs[0] = Q[obs, action]
        t = -1
        tau = -1
        T = np.inf

        while not done or t != T-1:
            t += 1
            if t < T:
                obs_prime, reward, done = env.step(action)
                states[(t+1)%n] = obs_prime
                if done:
                    T = t+1
                    deltas[t%n] = reward - Qs[t%n]
                else:
                    deltas[t%n] = reward + gamma * \
                    np.sum([policy_proba(policy, obs_prime, i, epsilon) * \
                    Q[obs_prime, i] for i in range(4)]) - Qs[t%n]
                    action = eps_greedy_policy(Q, obs_prime, epsilon, \
                                               env.action_space_size)
                    Qs[(t+1)%n] = Q[obs_prime, action]
                    pis[(t+1)%n] = policy_proba(policy, obs_prime, action, epsilon)
            tau = t-n+1
            if tau > -1:
                Z = 1
                G = Qs[tau%n]
                for k in range(tau,min(tau+n-1, T-1)):
                    G += Z*deltas[k%n]
                    Z *= gamma * Z * pis[(k+1)%n]
                s = states[tau%n]
                a = actions[tau%n]
                # Update state-action value function.
                Q[s,a] += alpha * (G - Q[s,a])
                # Make policy greedy w.r.t. Q.
                action_values = [Q[s,i] for i in range(4)]
                policy[s] = np.argmax(action_values)
        epsilon = decay(epsilon)
        if episode % 100 == 0:
            print_episode(episode,n_episodes)
    print_episode(n_episodes, n_episodes)
    return policy


if __name__ == '__main__':
    n = 4
    alpha = 0.01
    gamma = 1
    epsilon = 1
    n_episodes = 1000
    env = GridWorld()
    policy = n_step_tree_backup(env, n , alpha, gamma, epsilon, n_episodes)
    test_policy(env, policy, 10)
