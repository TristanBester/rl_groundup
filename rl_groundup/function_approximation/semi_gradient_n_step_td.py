# Created by Tristan Bester.
import sys
import numpy as np
sys.path.append('../')
from utils import print_episode
from functions import LinearValueFunction

'''
n-Step semi-gradient TD for estimating the
state-value function for a given policy.
Algorithm available on page 171 of
"Reinforcement Learning: An Introduction."

Book reference:
Sutton, R. and Barto, A., 2014. Reinforcement Learning:
An Introduction. 1st ed. London: The MIT Press.
'''


def semi_gradient_n_step_td(env, policy, n, alpha, gamma, n_episodes, tile_coder):
    # Initialization.
    v = LinearValueFunction(tile_coder.total_n_tiles)
    states = [None] * n
    rewards = np.zeros(n)

    for episode in range(n_episodes):
        done = False
        obs = env.reset()
        states[0] = tile_coder.get_tile_code(obs)
        t = 0
        tau = -1
        T = np.inf

        while not done or tau != T-1:
            if t < T:
                feature_vectors = tile_coder.get_feature_vectors_for_actions(obs, \
                                  env.action_space_size)
                a = policy.greedy_action(feature_vectors)
                obs, reward, done = env.step(a)
                states[(t+1)%n] = tile_coder.get_tile_code(obs)
                rewards[(t+1)%n] = reward
                if done:
                    T = t+1
            tau = t-n+1
            if tau > -1:
                # Calculate n-step return.
                G = np.sum([gamma**(i-tau-1)*rewards[i%n] for i in range(tau+1, \
                            min(tau+n, T))])
                if tau+n < T:
                    G += gamma ** n * v.evaluate(states[(tau+n)%n])
                # Update weights.
                v.weights += alpha * np.dot((G-v.evaluate(states[tau%n])), \
                                             states[tau%n])
            t += 1
        print_episode(episode, n_episodes)
    print_episode(n_episodes, n_episodes)
    return v
