# Created by Tristan Bester.
import sys
import gym
sys.path.append('../')
import numpy as np
from itertools import product
from utils import create_surface_plot, print_episode

'''
First-visit Monte Carlo prediction used to evaluate the policy defined on page
77 in the blackjack environment defined on page 76 of "Reinforcement Learning:
An Introduction."
Algorithm available on page 76.

Book reference:
Sutton, R. and Barto, A., 2014. Reinforcement Learning:
An Introduction. 1st ed. London: The MIT Press.
'''


def mc_pred(env, policy, n_episodes):
    '''First-visit Monte-Carlo prediction algorithm.'''
    hands = range(12, 22)
    dealer = range(1, 11)
    usable = [True, False]
    obs_space = product(hands, dealer, usable)

    # Initialization.
    keys = list(obs_space)
    V = dict.fromkeys(keys, 0)
    returns = {key:[] for key in keys}

    # For all hands less than 12 the player will hit to attain a hand in
    # the interval [12, 21]. Function prevents these states from being tracked
    # as optimal action already known.
    is_valid = lambda x: True if x[0] > 11 and x[0] < 22 else False

    for episode in range(n_episodes):
        done = False
        obs = env.reset()
        states = []
        if is_valid(obs):
            states.append(obs)

        # Generate an episode using policy.
        while not done:
            action = policy[obs]
            obs, reward, done, info = env.step(action)
            if obs not in states and is_valid(obs):
                states.append(obs)

        # Append return that follows first occurance of each state visited.
        for state in states:
            ls = returns[state]
            returns[state].append(reward)

        # Updated action-value function.
        for state,G in returns.items():
            if len(G) > 0:
                V[state] = np.mean(G)

        if episode % 1000 == 0:
            print_episode(episode, n_episodes)
    print_episode(n_episodes, n_episodes)
    return V


def plot_value_functions(V):
    '''Plot a value function for then the player does and does not
    have a usable ace in their hand.
    '''
    keys = np.array(list(V.keys()))
    keys_u = keys[keys[:,2] == 1]
    keys_n = keys[keys[:,2] != 1]
    z_u = [V[tuple(s)] for s in keys_u]
    z_n = [V[tuple(s)] for s in keys_n]
    create_surface_plot(keys_u[:, 0], keys_u[:, 1], z_u, \
                        "Hand:","Dealer:","Usable ace:")
    create_surface_plot(keys_n[:, 0], keys_n[:, 1], z_n, \
                        "Hand:","Dealer:","No usable ace:")


if __name__ == '__main__':
    env = gym.make('Blackjack-v0')
    n_episodes = 100000

    # Initialization of policy to be evaluated.
    possible_states = list(product(range(4,32), range(1,11), [True, False]))
    policy = dict((key, 0) if key[0] >= 20 else (key, 1) for key in possible_states)

    V = mc_pred(env, policy, n_episodes)
    plot_value_functions(V)
    env.close()
