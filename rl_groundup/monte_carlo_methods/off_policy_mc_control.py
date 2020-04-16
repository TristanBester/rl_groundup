import sys
import gym
import numpy as np
sys.path.append('../')
from utils import print_episode
from itertools import product, tee
from fv_mc_prediction import mc_pred, plot_value_functions


def off_policy_mc(env, gamma, b_policy, n_episodes):
    # Create required iterators.
    n_hands, n_dealer, usable = tuple([env.observation_space[i].n for i in range(3)])
    state_space = product(range(n_hands), range(n_dealer), [True, False])
    it_states1, it_states2 = tee(state_space)
    action_space = range(2)
    sa_pairs = product(it_states1, action_space)
    it_pairs1, it_pairs2 = tee(sa_pairs)

    # Initialization
    Q = dict.fromkeys(it_pairs1, 0.0)
    C = dict.fromkeys(it_pairs2, 0.0)
    target = dict.fromkeys(it_states2, 0)

    # Solving for optimal policy.
    for episode in range(n_episodes):
        if episode % 10000 == 0:
            print_episode(episode, n_episodes)
        done = False
        obs = env.reset()
        states = []
        actions = []
        rewards = []

        while not done:
            action = b_policy(obs)
            states.append(obs)
            obs, reward, done, info = env.step(action)
            actions.append(action)
            rewards.append(reward)

        G = 0
        W = 1

        for t in range(len(states)-1, -1, -1):
            G = gamma * G + rewards[t]
            s,a = states[t], actions[t]
            C[(s,a)] += W
            Q[(s,a)] += (W/C[(s,a)])*(G - Q[(s,a)])
            action_values = [Q[s,i] for i in range(env.action_space.n)]
            target[(s,a)] = np.argmax(action_values)
            if a == target[(s,a)]:
                W *= (1/0.5)
            else:
                break
    print_episode(n_episodes, n_episodes)
    return target


if __name__ == '__main__':
    n_episodes_control = 3000000
    n_episodes_prediction = 100000
    gamma = 0.99
    env = gym.make('Blackjack-v0')
    b_policy = lambda x: np.random.randint(2)
    print('Beginning control...\n')
    target = off_policy_mc(env, gamma, b_policy, n_episodes_control)
    print('Beginning prediction...\n')
    V = mc_pred(env, target, n_episodes_prediction)
    plot_value_functions(V)
    env.close()
