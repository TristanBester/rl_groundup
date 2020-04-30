# Created by Tristan Bester.
import sys
import numpy as np
sys.path.append('../')
from envs import GridWorld
from itertools import product
from utils import print_episode, test_policy

'''
Off-policy n-step Sarsa used to estimate the optimal policy for
the gridworld environment defined on page 48 of
"Reinforcement Learning: An Introduction."
Algorithm available on page 122.

Book reference:
Sutton, R. and Barto, A., 2014. Reinforcement Learning:
An Introduction. 1st ed. London: The MIT Press.
'''


def eps_greedy_proba(policy, s, a, epsilon):
    '''Return the probability of the given epsilon-greedy policy
    taking the specified action in the specified state.'''
    if policy[s] == a:
        return (epsilon/4) + (1-epsilon)
    else:
        return epsilon/4


def off_policy_n_step_sarsa(env, n, alpha, gamma, epsilon, n_episodes):
    # Initialize target policy and state-action value function.
    sa_pairs = product(range(env.observation_space_size), \
                       range(env.action_space_size))
    Q = dict.fromkeys(sa_pairs, 0.0)
    policy = dict.fromkeys(range(env.observation_space_size), 0)
    states = np.zeros(n)
    actions = np.zeros(n)
    rewards = np.zeros(n)

    decay = lambda x: x - 2/n_episodes if x-2/n_episodes > 0.1 else 0.1

    for episode in range(n_episodes):
        done = False
        obs = env.reset()
        action = np.random.randint(4)
        states[0] = obs
        actions[0] = action
        t = 0
        tau = -1
        T = np.inf

        while not done or tau != T-1:
            if t < T:
                obs_prime, reward, done = env.step(action)
                states[(t+1)%n] = obs_prime
                rewards[(t+1)%n] = reward
                if done:
                    T = t+1
                else:
                    action = np.random.randint(4)
                    actions[(t+1)%n] = action
            tau = t-n+1
            if tau > -1:
                p = 1
                for i in range(tau+1,min(tau+n-1, T-1)):
                    s = states[i%n]
                    a = actions[i%n]
                    policy_proba = eps_greedy_proba(policy, s, a, epsilon)
                    # 0.25 constant used as behaviour policy acts randomly.
                    p *= policy_proba/0.25
                G = np.sum([gamma**(i-tau-1)*rewards[i%n] for i in \
                            range(tau+1, min(tau+n,T))])
                if tau + n < T:
                    s = states[(tau+n)%n]
                    a = actions[(tau+n)%n]
                    G += gamma ** n * Q[s,a]
                s = states[tau%n]
                a = actions[tau%n]
                # Update state-action value estimate of the target policy.
                Q[s,a] += alpha * p * (G - Q[s,a])
                # Make target policy greedy w.r.t. Q.
                action_values = [Q[s, i] for i in range(4)]
                policy[s] = np.argmax(action_values)
            t += 1
        epsilon = decay(epsilon)
        if episode % 1000 == 0:
            print_episode(episode, n_episodes)
    print_episode(n_episodes, n_episodes)
    return policy


if __name__ == '__main__':
    n = 4
    alpha  = 0.001
    gamma = 1
    epsilon = 1
    n_episodes = 10000
    n_tests = 10
    env = GridWorld()
    print('Beginning control...\n')
    policy = off_policy_n_step_sarsa(env, n, alpha, gamma, epsilon, n_episodes)
    test_policy(env, policy, n_tests)
