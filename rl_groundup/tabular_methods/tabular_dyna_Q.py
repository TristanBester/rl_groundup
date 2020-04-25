import sys
import time
import numpy as np
sys.path.append('../')
from envs import Maze
from itertools import product, tee
from utils import print_episode, print_grid_world_actions


def eps_greedy_policy(Q, s, epsilon):
    if np.random.uniform() < epsilon:
        return np.random.randint(4)
    else:
        action_values = [Q[s, i] for i in range(4)]
        return np.argmax(action_values)


def tabular_dyna_Q(env, alpha, gamma, epsilon, n_episodes, n):
    sa_pairs = product(range(env.observation_space_size), \
                       range(env.action_space_size))
    pairs_one, pairs_two = tee(sa_pairs)
    Q = dict.fromkeys(pairs_one, 0)
    model = {pair:(-1,-1) for pair in pairs_two}

    for episode in range(n_episodes):
        done = False
        obs = env.reset()
        while not done:
            action = eps_greedy_policy(Q, obs, epsilon)
            obs_prime, reward, done = env.step(action)
            max_Q = np.argmax([Q[obs_prime, i] for i in range(4)])
            Q[obs, action] += alpha * (reward + gamma * Q[obs_prime, max_Q] - Q[obs, action])
            model[obs, action] = (reward, obs_prime)
            obs = obs_prime

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







def create_greedy_policy(env, Q):
    policy = {}
    for s in range(env.observation_space_size):
        action_values = [Q[s, i] for i in range(env.action_space_size)]
        policy[s] = np.argmax(action_values)
    return policy


def test_policy(env, policy, n_tests):
    input('Press any key to begin tests.')
    for i in range(n_tests):
        done = False
        obs = env.reset()
        env.render()
        time.sleep(0.3)
        while not done:
            a = policy[obs]
            obs, _, done = env.step(a)
            env.render()
            time.sleep(0.3)


if __name__ == '__main__':
    n = 25
    alpha = 0.1
    gamma = 0.95
    epsilon = 0.1
    n_episodes = 100
    n_tests = 10
    env = Maze()
    Q = tabular_dyna_Q(env, alpha, gamma, epsilon, n_episodes, n)
    policy = create_greedy_policy(env, Q)
    test_policy(env, policy, n_tests)
