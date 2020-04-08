import sys
import numpy as np
sys.path.append('../')
from utils import GridWorld, create_value_func_plot

def policy(state):
    return np.random.randint(0,4)

def TD(env, policy, n_steps, alpha, gamma, n_episodes):
    n_states = env.observation_space_size
    V = np.zeros(n_states)

    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        T = np.inf
        tau = 0
        t = 0

        states = np.zeros(n)
        rewards = np.zeros(n)
        states[t] = obs

        while not done or tau != T - 1:
            if t < T:
                action = policy(obs)
                obs, reward, done = env.step(action)
                states[(t+1) % n] = obs
                rewards[(t+1) % n] = reward
                if done:
                    T = t + 1
            tau = t - n + 1
            if tau >= 0:
                G = 0
                for i in range(tau + 1, min(tau+n, T) + 1):
                    G += gamma ** (i-tau-1) * rewards[i % n]
                if tau + n < T:
                    state = int(states[(tau + n) % n])
                    G += gamma ** n * V[state]
                state = int(states[tau % n])
                V[state] += alpha * (G - V[state])
            t += 1
    return V



if __name__ == '__main__':
    env = GridWorld()
    n = 4
    alpha = 0.1
    n_episodes = 50000
    gamma = 0.95
    V = TD(env, policy, n, alpha, gamma, n_episodes)
    create_value_func_plot(V, (4,4))
