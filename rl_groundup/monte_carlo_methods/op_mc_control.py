import sys
import gym
sys.path.append('../')
import numpy as np
from itertools import product
from utils import print_episode
from fv_mc_prediction import mc_pred, plot_value_functions

def op_mc_control(env, n_episodes):
    obs_space = product(range(1,32), range(1,11), [True, False])
    states = list(obs_space)
    sa_pairs = product(states, range(2))
    keys = list(sa_pairs)

    # Initization.
    Q = {s:np.zeros((2)) for s in states}
    returns = {pair:[] for pair in keys}
    policy = {s[0]:1 for s in keys}
    epsilon = 1.0

    for episode in range(n_episodes):
        done = False
        obs = env.reset()
        pairs = []

        while not done:
            action = policy[obs]
            pairs.append([obs,action])
            obs, reward, done, info = env.step(action)
        pairs.append((obs, policy[obs]))

        for s,a in pairs:
            returns[s,a].append(reward)

        for (s,a),G in returns.items():
            if len(G) > 0:
                Q[s][a] = np.mean(G)

        for s,_ in pairs:
            opt_a = np.argmax(Q[s])
            if np.random.uniform() < epsilon:
                policy[s] = env.action_space.sample()
            else:
                policy[s] = opt_a

        epsilon = epsilon - 3/n_episodes if epsilon > 0.1 else 0.1

        if episode % 1000 == 0:
            print_episode(episode, n_episodes)
    print_episode(n_episodes, n_episodes)
    return policy



if __name__ == '__main__':
    n_episodes = 100000
    env = gym.make('Blackjack-v0')
    print('Beginning control...\n')
    policy = op_mc_control(env, n_episodes)
    print('Beginning prediction...\n')
    V = mc_pred(env, policy, n_episodes)
    plot_value_functions(V)
    env.close()
