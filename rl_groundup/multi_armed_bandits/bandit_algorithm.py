import sys
import numpy as np
sys.path.append('../')
from envs import KArmedBandit
import matplotlib.pyplot as plt

n_episodes = 1000
K = 4
Q = np.zeros(K)
N = np.zeros(K)
epsilon = 0.1

bandit = KArmedBandit(K)
selections = []

for episode in range(n_episodes):
    if np.random.random() < epsilon:
        action = np.random.choice(range(K))
    else:
        action = np.argmax(Q)
    reward = bandit.pull_arm(action)
    selections.append(N.copy())
    N[action] += 1
    Q[action] += (1/N[action]) * (reward - Q[action])

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
ax[1].bar(range(K), Q[:])
ax[1].set_xlabel('Arm index:')
ax[1].set_ylabel('Arm value:')
ax[1].set_title('Value of each arm:')
ax[2].bar(range(K), N[:])
ax[2].set_xlabel('Arm index:')
ax[2].set_ylabel('Total times arm pulled:')
ax[2].set_title('Number of times each arm pulled:')
plt.show()
