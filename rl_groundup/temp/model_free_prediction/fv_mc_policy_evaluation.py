import sys
import gym
import numpy as np
sys.path.append('../')
from utils import create_surface_plot

def policy(hand, usable):
    if hand < 15:
        return 1
    elif hand < 17 and usable:
        return 1
    else:
        return 0

def add_state(dct,state):
    if state not in dct:
        dct[state] = [1,0]
    else:
        dct[state][0] += 1

    return dct

def get_state_values(value_func):
    states = np.array(list(value_func.keys()))
    counters = np.array(list(value_func.values()))
    values = counters[:,1] / counters[:,0]
    return states, values


def fv_mc_eval(env, policy, n_episodes):
    usable = {}
    no_usable = {}

    for i in range(n_episodes):
        done  = False
        obs = env.reset()
        fin_reward = 0
        episode_states = [obs]

        while not done:
            action = policy(obs[0], obs[2])
            obs, reward, done, info = env.step(action)
            # Save only first visit to state in episode.
            if obs not in episode_states:
                episode_states.append(obs)
            fin_reward = reward

        for state in episode_states:
            if state[2]:
                add_state(usable, state)
                usable[state][1] += fin_reward
            else:
                add_state(no_usable, state)
                no_usable[state][1] += fin_reward

    states_u, values_u = get_state_values(usable)
    states_n, values_n = get_state_values(no_usable)
    return states_u, states_n, values_u, values_n



if __name__ == '__main__':
    n_episodes = 100000
    env = gym.make('Blackjack-v0')
    states_u, states_n, values_u, values_n = fv_mc_eval(env, policy, n_episodes)

    x_label = 'Players hand:'
    y_label = 'Dealers hand:'
    title_u = 'Player has usable ace:'
    title_n = 'Player does not have a usable ace:'

    create_surface_plot(x = states_u[:,0], y = states_u[:,1], z = values_u, \
                        x_label=x_label, y_label=y_label, title=title_u)
    create_surface_plot(x = states_n[:,0], y = states_n[:,1], z = values_n, \
                        x_label=x_label, y_label=y_label, title=title_n)
    env.close()
