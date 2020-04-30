# Created by Tristan Bester.
import sys
import numpy as np
sys.path.append('../')
from envs import KArmedBandit
import matplotlib.pyplot as plt

'''
Solution to the K-armed bandit problem. A simple bandit algorithm has been
used to solve the problem.
Algoithm available on page 24 of "Reinforcement Learning: An Introduction."
Book reference:
Sutton, R. and Barto, A., 2014. Reinforcement Learning:
An Introduction. 1st ed. London: The MIT Press.
'''

n_episodes = 1000
K = 4
epsilon = 0.1
bandit = KArmedBandit(K)

# Store data for plots.
selections = []

# Initialization.
Q = np.zeros(K)
N = np.zeros(K)

# Solving the problem.
for episode in range(n_episodes):
    if np.random.random() < epsilon:
        action = np.random.choice(range(K))
    else:
        action = np.argmax(Q)
    reward = bandit.pull_arm(action)
    selections.append(N.copy())
    N[action] += 1
    Q[action] += (1/N[action]) * (reward - Q[action])

# Plot the results.
selections = np.array(selections)
fig, ax = plt.subplots(1, 3)
fig.set_size_inches((13,4))
ax[0].plot(range(n_episodes), selections[:, 0], label='Arm zero')
ax[0].plot(range(n_episodes), selections[:, 1], label='Arm one')
ax[0].plot(range(n_episodes), selections[:, 2], label='Arm two')
ax[0].plot(range(n_episodes), selections[:, 3], label='Arm three')
ax[0].set_xlabel('Episode number:')
ax[0].set_ylabel('Number of times arm chosen:')
ax[0].set_title('Number of times each arm was pulled:')
ax[0].legend()
ax[1].bar(range(K), bandit.arm_probas)
ax[1].set_xlabel('Arm index:')
ax[1].set_ylabel('Probability:')
ax[1].set_title('Arm reward probability:')
ax[2].bar(range(K), N[:])
ax[2].set_xlabel('Arm index:')
ax[2].set_ylabel('Total times arm pulled:')
ax[2].set_title('Number of times each arm pulled:')
plt.show()
