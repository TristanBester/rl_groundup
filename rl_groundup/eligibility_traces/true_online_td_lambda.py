# Created by Tristan Bester.
import sys
import numpy as np
sys.path.append('../')
from envs import RandomWalk
from functions import LinearValueFunction
from utils import encode_state, create_bar_plot

'''
True Online TD lambda used to estimate the value function
in a nineteen-state Random Walk MRP. An MRP was used as this
algorithm is focused on the prediction problem and not the
control problem. Therefore, as we are dealing with the prediction
problem there is no need to distinguish the dynamics due to the
environment from those due to the agent. More information
justifying this decision can be found on page 102
of "Reinforcement Learning: An Introduction.". The algorithm
can also be found on page 246 of the same text.

Book reference:
Sutton, R. and Barto, A., 2014. Reinforcement Learning:
An Introduction. 1st ed. London: The MIT Press.
'''


def online_td_lambda(env, lamda, alpha, gamma, n_episodes):
    # Initialize value function.
    v = LinearValueFunction(env.n_states)

    for episode in range(n_episodes):
        done = False
        obs = env.reset()
        obs_vec  = encode_state(obs, env.n_states)
        z = np.zeros(env.n_states)
        V_old = 0

        while not done:
            obs_prime, reward, done = env.step()
            obs_prime_vec = encode_state(obs_prime, env.n_states)
            V = v.evaluate(obs_vec)
            V_prime = v.evaluate(obs_prime_vec)
            delta = reward + gamma * V_prime - V
            # Update eligibility traces.
            z = gamma * lamda * z + (1 - alpha * gamma * lamda * np.dot(z, obs_vec)) * obs_vec
            # Update weights.
            v.weights += alpha * (delta + V - V_old) * z - alpha * (V - V_old) * obs_vec
            V_old = V_prime
            obs_vec = obs_prime_vec
    return v


if __name__ == '__main__':
    alpha = 0.1
    gamma = 1
    lamda = 0.5
    n_episodes = 100
    env = RandomWalk(19)
    v = online_td_lambda(env, lamda, alpha, gamma, n_episodes)
    create_bar_plot(range(19), v.weights, 'State index:', 'State value:', \
                    'Approximate state values in Random Walk MRP:')
