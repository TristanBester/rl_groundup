# Created by Tristan Bester.
import sys
import numpy as np
sys.path.append('../')
from envs import KArmedBandit
import matplotlib.pyplot as plt

'''
Solution to the K-armed bandit problem. Upper-Confidence-Bound Action
Selection has been used to solve the problem.
Information available on page 26 of "Reinforcement Learning: An Introduction."
Book reference:
Sutton, R. and Barto, A., 2014. Reinforcement Learning:
An Introduction. 1st ed. London: The MIT Press.
'''


def action_potential(a,t):
    '''Calulate the potential of the action being optimal.'''
    if N[a] == 0:
        return np.inf
    else:
        return Q[a] + c * np.sqrt((np.log(t)/N[a]))


K = 4
c = 2
n_time_steps = 1000
bandit = KArmedBandit(K)

# Store data for plots.
selections = []

# Initialization.
Q = np.zeros(K)
N = np.zeros(K)

# Solve the problem.
for t in range(n_time_steps):
        potentials = [action_potential(a,t) for a in range(K)]
        action = np.argmax(potentials)
        reward = bandit.pull_arm(action)
        N[action] += 1
        Q[action] += (1/N[action]) * (reward - Q[action])
        selections.append(N.copy())

# Display the results.
selections = np.array(selections)
fig, ax = plt.subplots(1,2)
fig.set_size_inches(10,4)
ax[0].plot(range(n_time_steps), selections[:, 0], label='Arm zero')
ax[0].plot(range(n_time_steps), selections[:, 1], label='Arm one')
ax[0].plot(range(n_time_steps), selections[:, 2], label='Arm two')
ax[0].plot(range(n_time_steps), selections[:, 3], label='Arm three')
ax[0].set_xlabel('Episode number:')
ax[0].set_ylabel('Number of times arm chosen:')
ax[0].set_title('Number of times each arm was pulled:')
ax[0].legend()
ax[1].bar(range(K), bandit.arm_probas[:])
ax[1].set_xlabel('Arm index:')
ax[1].set_ylabel('Probability:')
ax[1].set_title('Reward probability of each arm:')
plt.show()
