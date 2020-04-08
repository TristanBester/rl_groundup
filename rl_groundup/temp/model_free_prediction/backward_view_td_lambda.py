import sys
import numpy as np
sys.path.append('../')
from utils import GridWorld, create_value_func_plot

def policy(state):
    return np.random.randint(0,4)

def TD_lambda(env, policy, alpha, gamma, lmbda, n_episodes):
    n_states = env.observation_space_size
    V = np.zeros(n_states)
    states = np.array(range(n_states))

    for episode in range(n_episodes):
        E = np.zeros(V.shape)
        obs = env.reset()
        done = False
        rewards = []

        while not done:
            action = policy(obs)
            n_obs, reward, done = env.step(action)
            td_error = reward + V[int(n_obs)] - V[int(obs)]
            E[int(obs)] += 1
            for state in states:
                V[state] += alpha * td_error * E[state]
                E[state] *= lmbda * gamma
            obs = n_obs

    return V

if __name__ == '__main__':
    env = GridWorld()
    alpha = 0.01
    gamma = 0.99
    lmbda = 0.5
    n_episodes = 10000
    V = TD_lambda(env, policy, alpha, gamma, lmbda, n_episodes)
    create_value_func_plot(V, (4,4))
