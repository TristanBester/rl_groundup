import sys
import numpy as np
sys.path.append('../')
from itertools import product
from envs import WindyGridWorld
from td_zero_prediction import td_pred
from utils import print_episode, create_value_func_plot


def eps_greedy_policy(Q, s, epsilon):
    action_values = [Q[s, i] for i in range(4)]
    if np.random.uniform() < epsilon:
        return np.random.randint(4)
    else:
        return np.argmax(action_values)

def sarsa(env, gamma, alpha, epsilon, n_episodes):
    sa_pairs = product(range(70), range(4))
    Q = dict.fromkeys(sa_pairs, 0.0)

    for episode in range(n_episodes):
        if episode % 1000 == 0:
            print_episode(episode, n_episodes)
        done = False
        obs = env.reset()
        action = eps_greedy_policy(Q, obs, epsilon)

        while not done:
            obs_prime, reward, done = env.step(action)
            action_prime = eps_greedy_policy(Q, obs_prime, epsilon)
            Q[obs,action] += alpha * (reward + gamma * \
                             (Q[obs_prime, action_prime]) - Q[obs, action])
            obs = obs_prime
            action = action_prime
        epsilon = epsilon - 4/n_episodes if epsilon - 2/n_episodes > 0 else 0.1

    print_episode(n_episodes, n_episodes)
    return Q

def create_greedy_policy(Q):
    policy = np.zeros(70)
    for i in range(70):
        action_values = [Q[i, a] for a in range(4)]
        policy[i] = np.argmax(action_values)
    return policy


if __name__ == '__main__':
    n_episodes_control = 500000
    n_episodes_prediction = 100000
    gamma = 0.99
    alpha = 0.1
    epsilon = 1.0
    env = WindyGridWorld()
    print('Beginning control...\n')
    Q = sarsa(env, gamma, alpha,epsilon, n_episodes_control)
    policy = create_greedy_policy(Q)
    print('Beginning prediction...\n')
    V =  td_pred(env, policy, alpha, gamma, n_episodes_prediction)
    create_value_func_plot(V, (7, 10), '')
