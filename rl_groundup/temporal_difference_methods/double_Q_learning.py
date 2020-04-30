# Created by Tristan Bester.
import sys
import numpy as np
sys.path.append('../')
from envs import RaceTrack
from itertools import product
from Q_learning import create_greedy_policy
from utils import print_episode, test_policy

'''
Double Q-learning used to estimate the optimal policy for the
racetrack environment defined on page 91 of
"Reinforcement Learning: An Introduction."
Algorithm available on page 111.

Book reference:
Sutton, R. and Barto, A., 2014. Reinforcement Learning:
An Introduction. 1st ed. London: The MIT Press.
'''


def double_Q_eps_greedy_policy(s, Q_1, Q_2, epsilon):
    '''Epsilon-greedy policy for double Q-learning.'''
    if np.random.uniform() < epsilon:
        return np.random.randint(9)
    else:
        action_vals_one = np.array([Q_1[s, i] for i in range(9)])
        action_vals_two = np.array([Q_2[s, i] for i in range(9)])
        action_values = action_vals_one + action_vals_two
        return np.argmax(action_values)


def double_Q(env, alpha, gamma, epsilon, n_episodes):
    # Initialize state-action value functions.
    Q_1 = {}
    Q_2 = {}
    curr_row = 0
    for row, col in env.state_space:
        for i in range(curr_row, curr_row + row):
            positions = product([i], range(col))
            velocities = product(range(-3, 1), range(-2, 3))
            states = product(positions, velocities)
            # Key: (((pos_x, pos_y), (dy, dx)), action)
            for pair in product(states, range(9)):
                Q_1[pair] = 0
                Q_2[pair] = 0
        curr_row += row

    decay = lambda i,x: x/(i+1)

    for episode in range(n_episodes):
        done = False
        obs = env.reset()
        while not done:
            a = double_Q_eps_greedy_policy(obs, Q_1, Q_2, epsilon)
            obs_prime, reward, done = env.step(a)
            # Update state-action value estimate.
            if np.random.uniform() < 0.5:
                action_vals = [Q_1[obs_prime, i] for i in range(9)]
                a_prime = np.argmax(action_vals)
                Q_1[obs, a] += alpha * (reward +gamma*Q_2[obs_prime, a_prime]\
                                        - Q_1[obs,a])
            else:
                action_vals = [Q_2[obs_prime, i] for i in range(9)]
                a_prime = np.argmax(action_vals)
                Q_2[obs, a] += alpha * (reward + gamma*Q_1[obs_prime, a_prime]\
                                        - Q_2[obs,a])
            obs = obs_prime
        epsilon = decay(episode, epsilon)
        if episode % 100 == 0:
            print_episode(episode, n_episodes)
    print_episode(n_episodes, n_episodes)
    Q = {s:i + x for s,i,x in zip(Q_1.keys(), Q_1.values(), Q_2.values())}
    return Q



if __name__ == '__main__':
    alpha = 0.1
    gamma = 0.9
    epsilon = 1.0
    n_episodes = 10000
    n_tests = 30
    env = RaceTrack()
    Q = double_Q(env, alpha, gamma, epsilon, n_episodes)
    policy = create_greedy_policy(env, Q)
    test_policy(env, policy, n_tests)
