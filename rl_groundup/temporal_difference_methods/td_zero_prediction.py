import sys
import numpy as np
sys.path.append('../')
from envs import WindyGridWorld
from utils import print_episode

def td_pred(env, policy, alpha, gamma, n_episodes):
    V = np.zeros(70)

    for episode in range(n_episodes):
        if episode % 1000 == 0:
            print_episode(episode, n_episodes)
        done = False
        obs = env.reset()
        while not done:
            action = policy[obs]
            obs_prime, reward, done = env.step(action)
            V[obs] += alpha * (reward + gamma * V[obs_prime] - V[obs])
            obs = obs_prime
    print_episode(n_episodes, n_episodes)
    return V
