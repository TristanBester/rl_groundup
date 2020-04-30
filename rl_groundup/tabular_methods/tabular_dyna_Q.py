# Created by Tristan Bester.
import sys
import numpy as np
sys.path.append('../')
from envs import Maze
from itertools import product, tee
from utils import print_episode, eps_greedy_policy, create_greedy_policy, \
                  test_policy

'''
Tabular Dyna-Q used to find an optimal policy for the maze environment described
on page 135 of "Reinforcement Learning: An Introduction."
Algorithm available on page 135.

Book reference:
Sutton, R. and Barto, A., 2014. Reinforcement Learning:
An Introduction. 1st ed. London: The MIT Press.
'''


def tabular_dyna_Q(env, alpha, gamma, epsilon, n_episodes, n):
    # Create iterators.
    sa_pairs = product(range(env.observation_space_size), \
                       range(env.action_space_size))
    pairs_one, pairs_two = tee(sa_pairs)

    # Initialize state-action value function and model.
    Q = dict.fromkeys(pairs_one, 0)
    model = {pair:(-1,-1) for pair in pairs_two}

    for episode in range(n_episodes):
        done = False
        obs = env.reset()
        while not done:
            # Acting, model-learning and direct RL.
            action = eps_greedy_policy(Q, obs, epsilon, env.action_space_size)
            obs_prime, reward, done = env.step(action)
            max_Q = np.argmax([Q[obs_prime, i] for i in range(4)])
            Q[obs, action] += alpha * (reward + gamma * Q[obs_prime, max_Q] - Q[obs, action])
            model[obs, action] = (reward, obs_prime)
            obs = obs_prime

            # Q-planning algorithm.
            for i in range(n):
                possible_pairs = [(s,a) for s,a in list(model.keys()) if \
                                  (model[s,a] != (-1,-1))]
                idx = np.random.choice(len(possible_pairs))
                pair = possible_pairs[idx]
                s = pair[0]
                a = pair[1]
                r, s_prime = model[s,a]
                max_Q = np.argmax([Q[s, x] for x in range(4)])
                Q[s,a] += alpha * (r + gamma * Q[s_prime, max_Q] - Q[s,a])
        print_episode(episode, n_episodes)
    print_episode(n_episodes, n_episodes)
    return Q


if __name__ == '__main__':
    n = 25
    alpha = 0.1
    gamma = 0.95
    epsilon = 0.1
    n_episodes = 100
    n_tests = 10
    env = Maze()
    Q = tabular_dyna_Q(env, alpha, gamma, epsilon, n_episodes, n)
    policy = create_greedy_policy(Q, env.observation_space_size, \
                                  env.action_space_size)
    test_policy(env, policy, n_tests)
