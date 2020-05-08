import sys
sys.path.append('../')
import numpy as np
from utils import print_episode
from functions import LinearValueFunction


def semi_gradient_td_zero(env, policy, alpha, gamma, n_episodes, tile_coder):
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
            v.weights += alpha * (np.dot((reward + gamma*v.evaluate(s_prime)- \
                                  v.evaluate(s)), s))
            obs = obs_prime
        print_episode(episode, n_episodes)
    print_episode(n_episodes, n_episodes)
    return v
