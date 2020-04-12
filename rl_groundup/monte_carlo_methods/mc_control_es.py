# Created by Tristan Bester.
import sys
import gym
sys.path.append('../')
import numpy as np
from itertools import product
from fv_mc_prediction import mc_pred, plot_value_functions
from utils import print_episode


def get_init_policy():
    '''Return initial policy used in Monte-Carlo ES algorithm.'''
    # Initialize policy to always hit. This ensures for all possible states
    # where player hand is less than 12 the player chooses the optimal action.
    # As is shown in the book, the only states we are interested in learning
    # are those where the players hand is greater than 11.
    possible_states = list(product(range(4,32), range(1,11), [True, False]))
    return dict.fromkeys(possible_states, 1)


def mc_control(env, n_episodes):
    '''Monte-Carlo control with Exploring Starts.'''
    # Create required iterators and lists.
    obs_space = product(range(12,22), range(1,11), [True, False])
    states = list(obs_space)
    sa_pairs = product(states, range(2))
    keys = list(sa_pairs)

    # Initization.
    Q = {s:np.zeros((2)) for s in states}
    returns = {pair:[] for pair in keys}
    starting_sa_pairs = list(returns.keys())
    policy = get_init_policy()

    # Don't track hands where optimal action is known (action = 1 if hand < 12).
    is_valid = lambda x: True if x[0] > 11 and x[0] < 22 else False
    player = lambda x: [x,0]

    for episode in range(n_episodes):
        env.reset()
        # Select random starting state and action.
        rand = np.random.randint(len(starting_sa_pairs))
        (x, y, usable),a = starting_sa_pairs[rand]
        # Configure the environment to use exploring starts.
        env.player = player(x)
        env.dealer = player(y)
        # Used to store result of episode.
        done = a == 0
        episode_data = [starting_sa_pairs[rand]] if not done else []
        obs = starting_sa_pairs[rand][0]

        # Query environment if hold chosen as starting action.
        if done:
            obs, reward, done, info = env.step(a)
            episode_data.append((obs,a))
        else:
            # Hit chosen as starting action.
            while not done:
                a = policy[obs]
                obs, reward, done, info = env.step(a)
                if obs not in episode_data and is_valid(obs):
                    episode_data.append((obs,a))

        # Append return that follows first occurance of each state,action pair.
        for obs, a in episode_data:
            returns[obs,a].append(reward)

        # Update action_value function.
        for pair, G in returns.items():
            if len(G) > 0:
                s,a = pair
                Q[s][a] = np.mean(G)

        # Update policy - make greedy w.r.t. action-value function.
        for s,ls in Q.items():
            policy[s] = np.argmax(ls)

        if episode % 1000 == 0:
            print_episode(episode, n_episodes)
    print_episode(n_episodes, n_episodes)
    return policy


if __name__ == '__main__':
    env = gym.make('Blackjack-v0')
    n_episodes_control = 1000
    n_episodes_prediction = 1000

    print('Starting control:\n')
    policy = mc_control(env, n_episodes_control)
    print('Starting prediction:\n')
    V = mc_pred(env, policy, n_episodes_prediction)
    plot_value_functions(V)
    env.close()
