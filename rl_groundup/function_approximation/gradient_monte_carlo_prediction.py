# Created by Tristan Bester.
import sys
import numpy as np
sys.path.append('../')
from utils import print_episode
from functions import LinearValueFunction

'''
Gradient Monte Carlo Algorithm for estimating the
state-value function for a given policy.
Algorithm available on page 165 of
"Reinforcement Learning: An Introduction."

Book reference:
Sutton, R. and Barto, A., 2014. Reinforcement Learning:
An Introduction. 1st ed. London: The MIT Press.
'''


def gradient_mc_prediction(env, policy, alpha, n_episodes, tile_coder):
    # Initialization.
    v = LinearValueFunction(tile_coder.total_n_tiles)
    states = []
    rewards = [None]

    for episode in range(n_episodes):
        done = False
        obs = env.reset()
        # Store the feature vector representation of the state.
        states.append(tile_coder.get_tile_code(obs))
        feature_vectors = tile_coder.get_feature_vectors_for_actions(obs, \
                          env.action_space_size)
        a = policy.greedy_action(feature_vectors)

        while not done:
            obs, reward, done = env.step(a)
            feature_vectors = tile_coder.get_feature_vectors_for_actions(obs, \
                              env.action_space_size)
            a = policy.greedy_action(feature_vectors)
            rewards.append(reward)
            states.append(tile_coder.get_tile_code(obs))

        for i in range(len(states)):
            G = np.sum(rewards[i+1:])
            # Update weights.
            v.weights += alpha * np.dot((G - v.evaluate(states[i])), states[i])
        print_episode(episode, n_episodes)
    print_episode(n_episodes, n_episodes)
    return v
