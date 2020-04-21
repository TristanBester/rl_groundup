import sys
import time
import numpy as np
sys.path.append('../')
from envs import RaceTrack
from itertools import product
from utils import print_episode, create_line_plot


def eps_greedy_policy(Q, s, epsilon):
    if np.random.uniform() < epsilon:
        return np.random.randint(9)
    else:
        action_values = [Q[s, i] for i in range(9)]
        return np.argmax(action_values)


def Q_learing(env, alpha, gamma, epsilon, n_episodes):
    rewards = []
    Q = {}
    curr_row = 0
    for row, col in env.state_space:
        for i in range(curr_row, curr_row + row):
            positions = product([i], range(col))
            velocities = product(range(-3, 1), range(-2, 3))
            states = product(positions, velocities)
            sa_pairs = product(states, range(9))
            for pair in sa_pairs:
                Q[pair] = 0
        curr_row += row

    decay = lambda x: x - 2/n_episodes if x - 2/n_episodes > 0 else 0.1

    for episode in range(n_episodes):
        done = False
        val = 0
        obs = env.reset()

        while not done:
            action = eps_greedy_policy(Q, obs, epsilon)
            obs_prime, reward, done = env.step(action)
            val += reward
            action_values = [Q[obs_prime,i] for i in range(9)]
            opt_a = np.argmax(action_values)
            Q[obs,action] += alpha * (reward + gamma * Q[obs_prime,opt_a] \
                             - Q[obs,action])
            obs = obs_prime
        epsilon = decay(epsilon)

        rewards.append(val)

        if episode % 10 == 0:
            print_episode(episode, n_episodes)
    print_episode(n_episodes, n_episodes)
    create_line_plot(range(len(rewards)), rewards, 'Episode number:', \
                    'Return:', 'Agent returns over training:')
    return Q


def create_greedy_policy(env, Q):
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


def test_policy(env, policy, n_tests):
    input('Press \'Enter\' to start tests.')
    for i in range(n_tests):
        done = False
        obs = env.reset()
        env.render()
        while not done:
            time.sleep(0.5)
            action = policy[obs]
            obs,_,_ = env.step(action)
            env.render()



if __name__ == '__main__':
    alpha = 0.1
    gamma = 0.9
    epsilon = 1.0
    n_episodes = 20000
    n_tests = 100
    env = RaceTrack()
    Q = Q_learing(env, alpha, gamma, epsilon, n_episodes)
    policy = create_greedy_policy(env, Q)
    test_policy(env, policy, n_tests)
