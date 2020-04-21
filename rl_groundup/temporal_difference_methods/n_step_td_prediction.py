import sys
import numpy as np
sys.path.append('../')
from itertools import product
from utils import print_episode, create_value_func_plot
from envs import WindyGridWorld

def n_step_td_pred(env, policy, n, alpha, gamma, n_episodes):
    V = np.zeros(env.observation_space_size)
    states = np.zeros(n)
    rewards = np.zeros(n)

    for episode in range(n_episodes):
        done = False
        obs = env.reset()
        states[0] = obs
        tau = -1
        t = 0
        T = np.inf

        while not done or tau != T-1:
            if t < T:
                action = policy[obs]
                #print(obs, action)
                #input()
                obs_prime, reward, done = env.step(action)
                states[(t+1)%n] = obs_prime
                rewards[(t+1)%n] = reward
                obs = obs_prime
                if done:
                    T = t + 1
            tau = t - n + 1
            if tau > -1:
                G = np.sum([gamma ** (i-tau-1) * rewards[i % n] for i in \
                            range(tau + 1, min(tau+n,T))])
                if tau + n < T:
                    state = int(states[(tau+n)%n])
                    G += gamma ** n * V[state]
                state = int(states[tau%n])
                V[state] += alpha * (G - V[state])
            t += 1

        if episode % 1000 == 0:
            print_episode(episode, n_episodes)
    print_episode(n_episodes, n_episodes)
    return V
