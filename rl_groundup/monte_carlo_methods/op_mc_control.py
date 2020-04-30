import sys
import gym
import numpy as np
sys.path.append('../')
from itertools import product
from fv_mc_prediction import mc_pred
from utils import print_episode, plot_blackjack_value_functions

'''
On-policy first-visit Monte Carlo control used to find the optimal policy
for the blackjack environment defined on page 76 of "Reinforcement Learning:
An Introduction."
Algorithm available on page 83.

Book reference:
Sutton, R. and Barto, A., 2014. Reinforcement Learning:
An Introduction. 1st ed. London: The MIT Press.
'''


def op_mc_control(env, n_episodes):
    '''On-policy first-visit Monte Carlo control algorithm.'''
    obs_space = product(range(1,32), range(1,11), [True, False])
    states = list(obs_space)
    sa_pairs = product(states, range(2))
    keys = list(sa_pairs)

    # Initialization.
    Q = {s:np.zeros((2)) for s in states}
    returns = {pair:[] for pair in keys}
    policy = {s[0]:1 for s in keys}
    epsilon = 1.0

    for episode in range(n_episodes):
        done = False
        obs = env.reset()
        pairs = []

        # Generate an episode.
        while not done:
            action = policy[obs]
            pairs.append([obs,action])
            obs, reward, done, info = env.step(action)
        pairs.append((obs, policy[obs]))

        # Store returns for each state-action pair visited.
        for s,a in pairs:
            returns[s,a].append(reward)

        # Average returns for each state-action pair.
        for (s,a),G in returns.items():
            if len(G) > 0:
                Q[s][a] = np.mean(G)

        # Update policy (epsilon-greedy w.r.t action-value function).
        for s,_ in pairs:
            opt_a = np.argmax(Q[s])
            if np.random.uniform() < epsilon:
                policy[s] = env.action_space.sample()
            else:
                policy[s] = opt_a

        # Decay epsilon.
        epsilon = epsilon - 3/n_episodes if epsilon > 0.1 else 0.1

        if episode % 1000 == 0:
            print_episode(episode, n_episodes)
    print_episode(n_episodes, n_episodes)
    return policy


if __name__ == '__main__':
    n_episodes_control = 100000
    n_episodes_prediction = 100000
    env = gym.make('Blackjack-v0')

    print('Beginning control...\n')
    policy = op_mc_control(env, n_episodes_control)
    print('Beginning prediction...\n')
    V = mc_pred(env, policy, n_episodes_prediction)
    plot_blackjack_value_functions(V)
    env.close()
