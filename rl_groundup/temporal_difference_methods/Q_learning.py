# Created by Tristan Bester.
import sys
import numpy as np
sys.path.append('../')
from envs import RaceTrack
from itertools import product
from utils import print_episode,create_line_plot,eps_greedy_policy,test_policy

'''
Q-learning (off-policy TD control) used to estimate the optimal policy for the
racetrack environment defined on page 91 of
"Reinforcement Learning: An Introduction."
Algorithm available on page 107.

Book reference:
Sutton, R. and Barto, A., 2014. Reinforcement Learning:
An Introduction. 1st ed. London: The MIT Press.
'''


def Q_learing(env, alpha, gamma, epsilon, n_episodes):
    # Initialize state-action value function.
    Q = {}
    curr_row = 0
    for row, col in env.state_space:
        for i in range(curr_row, curr_row + row):
            positions = product([i], range(col))
            velocities = product(range(-3, 1), range(-2, 3))
            states = product(positions, velocities)
            sa_pairs = product(states, range(9))
            # Key: (((pos_x, pos_y), (dy, dx)), action)
            for pair in sa_pairs:
                Q[pair] = 0
        curr_row += row

    # Store rewards for plot.
    rewards = []
    decay = lambda x: x - 2/n_episodes if x - 2/n_episodes > 0 else 0.1

    for episode in range(n_episodes):
        done = False
        val = 0
        obs = env.reset()

        while not done:
            action = eps_greedy_policy(Q, obs, epsilon, env.action_space_size)
            obs_prime, reward, done = env.step(action)
            val += reward
            action_values = [Q[obs_prime,i] for i in range(9)]
            opt_a = np.argmax(action_values)
            # Update state-action value estimate.
            Q[obs,action] += alpha * (reward + gamma * Q[obs_prime,opt_a] \
                             - Q[obs,action])
            obs = obs_prime
        epsilon = decay(epsilon)
        rewards.append(val)
        if episode % 10 == 0:
            print_episode(episode, n_episodes)
    print_episode(n_episodes, n_episodes)

    # Plot rewards over training process.
    create_line_plot(range(len(rewards)), rewards, 'Episode number:', \
                    'Return:', 'Agent returns over training:')
    return Q


def create_greedy_policy(env, Q):
    '''Create policy that acts greedily w.r.t. Q. Method implementation specific
    to racetrack environment, thus implementation in /utils/policies.py not used.
    '''
    policy = {}
    curr_row = 0
    for row, col in env.state_space:
        for i in range(curr_row, curr_row + row):
            positions = product([i], range(col))
            velocities = product(range(-3, 1), range(-2, 3))
            states = product(positions, velocities)
            for state in states:
                action_values = [Q[state, i] for i in range(9)]
                policy[state] = np.argmax(action_values)
        curr_row += row
    return policy


if __name__ == '__main__':
    alpha = 0.1
    gamma = 0.9
    epsilon = 1.0
    n_episodes = 20000
    n_tests = 30
    env = RaceTrack()
    Q = Q_learing(env, alpha, gamma, epsilon, n_episodes)
    policy = create_greedy_policy(env, Q)
    test_policy(env, policy, n_tests)
