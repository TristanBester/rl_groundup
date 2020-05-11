# Created by Tristan Bester.
import sys
sys.path.append('../')
import numpy as np
from utils import print_episode
from functions import LinearValueFunction

'''
Semi-gradient TD(0) for estimating the
state-value function for a given policy.
Algorithm available on page 166 of
"Reinforcement Learning: An Introduction."

Book reference:
Sutton, R. and Barto, A., 2014. Reinforcement Learning:
An Introduction. 1st ed. London: The MIT Press.
'''


def semi_gradient_td_zero(env, policy, alpha, gamma, n_episodes, tile_coder):
    # Initialization.
    v = LinearValueFunction(tile_coder.total_n_tiles)

    for episode in range(n_episodes):
        done = False
        obs = env.reset()
        while not done:
            feature_vectors = tile_coder.get_feature_vectors_for_actions(obs, \
                              env.action_space_size)
            a = policy.greedy_action(feature_vectors)
            obs_prime, reward, done = env.step(a)
            s = tile_coder.get_tile_code(obs)
            s_prime = tile_coder.get_tile_code(obs_prime)
            # Update weights.
            v.weights += alpha * (np.dot((reward + gamma*v.evaluate(s_prime)- \
                                  v.evaluate(s)), s))
            obs = obs_prime
        print_episode(episode, n_episodes)
    print_episode(n_episodes, n_episodes)
    return v
