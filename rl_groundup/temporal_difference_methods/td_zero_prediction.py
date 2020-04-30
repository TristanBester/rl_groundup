# Created by Tristan Bester.
import sys
import numpy as np
sys.path.append('../')
from envs import WindyGridWorld
from utils import print_episode

'''
Tabular TD(0) used to estimate the state-value function for a given policy
in the windy gridworld environment defined on page 106 of
"Reinforcement Learning: An Introduction."
Algorithm available on page 98.

Book reference:
Sutton, R. and Barto, A., 2014. Reinforcement Learning:
An Introduction. 1st ed. London: The MIT Press.
'''


def td_pred(env, policy, alpha, gamma, n_episodes):
    # Initialize state-value function.
    V = np.zeros(70)

    for episode in range(n_episodes):
        done = False
        obs = env.reset()
        while not done:
            action = policy[obs]
            obs_prime, reward, done = env.step(action)
            # Update state-value estimate.
            V[obs] += alpha * (reward + gamma * V[obs_prime] - V[obs])
            obs = obs_prime
        if episode % 1000 == 0:
            print_episode(episode, n_episodes)
    print_episode(n_episodes, n_episodes)
    return V
