# Created by Tristan Bester.
import sys
import numpy as np
sys.path.append('../')
from envs import KArmedBandit
from utils import create_line_plot

'''
Solution to the K-armed bandit problem. The gradient bandit algorithm has been
used to solve the problem.
Information available on page 28 of "Reinforcement Learning: An Introduction."
Book reference:
Sutton, R. and Barto, A., 2014. Reinforcement Learning:
An Introduction. 1st ed. London: The MIT Press.
'''


def action_probabilites(H):
    '''Calculate the soft-max distribution of action preferences.'''
    P = np.zeros(10)
    for a in range(10):
        P[a] = np.exp(H[a])/(np.sum([np.exp(H[i]) for i in range(10)]))
    return P


n_time_steps = 1000
K = 10
alpha = 0.1

# Store data for plots.
reward_aves = []
rewards = []
opt_action_count = 0
opt = [0]

# Initialization.
H = np.zeros(K)

bandit = KArmedBandit(K)

# Solve the problem.
for t in range(1, n_time_steps+1):
    P = action_probabilites(H)
    action = np.random.choice(range(10), p=P)
    reward = bandit.pull_arm(action)
    rewards.append(reward)
    reward_aves.append(np.mean(rewards))
    if action == np.argmax(bandit.arm_probas):
        opt_action_count += 1
    opt.append(opt_action_count)

    for a in range(10):
        if a == action:
            H[a] = H[a] + alpha * (reward - np.mean(rewards)) * (1-P[a])
        else:
            H[a] = H[a] - alpha * (reward - np.mean(rewards)) * P[a]


# Plot the results.
title = 'Average reward:'
x_label = 'Time step:'
y_label = 'Average reward:'
create_line_plot(range(len(reward_aves)), reward_aves[:],
                 x_label, y_label, title)

title = 'Optimal actions taken over time:'
x_label = 'Time step:'
y_label = 'Optimal action count:'
create_line_plot(range(len(opt)), opt, x_label, y_label, title)
