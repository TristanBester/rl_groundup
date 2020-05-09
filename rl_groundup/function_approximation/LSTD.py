import sys
import numpy as np
sys.path.append('../')
from utils import print_episode, TileCoding

from functions import LinearPolicy
from envs import MountainCar
from functions import LinearValueFunction

'''
https://www.aaai.org/Papers/AAAI/2006/AAAI06-057.pdf
'''

def least_squares_td(env, policy, epsilon, alpha, gamma, n_episodes, tile_coder):
    n = tile_coder.total_n_tiles
    A = (1/epsilon) * np.eye(n)
    b = np.zeros((n,1))
    d = np.zeros((n,1))

    for episode in range(n_episodes):
        done = False
        obs = env.reset()


        while not done:
            feature_vectors = tile_coder.get_feature_vectors_for_actions(obs, \
                              env.action_space_size)
            a = policy.greedy_action(feature_vectors)
            obs_prime, reward, done = env.step(a)
            x = np.array(tile_coder.get_tile_code(obs)).reshape(-1,1)
            x_prime = np.array(tile_coder.get_tile_code(obs_prime)).reshape(-1,1)
            b = b + reward * x
            d = (x - gamma * x_prime)
            A = A + x @ d.T
            if env.steps == 2:
                inv_A = np.linalg.inv(A)
            else:
                t = np.eye(n) - (((x @ d.T)/(1+((d.T @ inv_A)@x)))@inv_A)
                inv_A = inv_A @ t
            theta = inv_A @ b
        print_episode(episode, n_episodes)
    print_episode(n_episodes, n_episodes)
    v = LinearValueFunction(tile_coder.total_n_tiles)
    v.weights = theta.flatten()
    return v
