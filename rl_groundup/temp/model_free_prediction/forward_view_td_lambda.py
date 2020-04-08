import sys
import numpy as np
sys.path.append('../')
from utils import GridWorld, create_value_func_plot

def policy(state):
    return np.random.randint(0,4)

def TD_lambda(env, policy, alpha, gamma, lmbda, n_episodes):
    n_states = env.observation_space_size
    V = np.zeros(n_states)

    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        states = [obs]
        rewards = []

        while not done:
            action = policy(obs)
            obs, reward, done = env.step(action)
            states.append(int(obs))
            rewards.append(reward)

        for i,state in enumerate(states):
            G = 0
            for x in range(1,i+1):
                G += np.sum(rewards[:x])
                G *= lmbda ** (x-1)
            G *= (1-lmbda)
            V[state] += alpha * (G - V[state])

    return V


if __name__ == '__main__':
    env = GridWorld()
    alpha = 0.1
    gamma = 0.99
    lmbda = 0.1
    n_episodes = 5000
    V = TD_lambda(env, policy, alpha, gamma, lmbda, n_episodes)
    create_value_func_plot(V, (4,4))
