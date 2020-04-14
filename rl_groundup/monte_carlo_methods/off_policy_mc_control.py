import sys
import numpy as np
sys.path.append('../')
from envs import RaceTrack
import time
from itertools import product
import pickle
from utils import print_episode
import os

def adjust_col(s):
    pos = s[0]
    y = pos[0]
    vel = s[1]

    if y == 0:
        s = ((y, pos[1] - 3), vel)
    elif y < 3:
        s = ((y, pos[1] - 2), vel)
    elif y == 3:
        s = ((y, pos[1] - 1), vel)
    elif y > 13 and y < 15:
        s = ((y, pos[1] - 1), vel)
    elif y > 14 and y < 23:
        s = ((y, pos[1] - 2), vel)
    elif y > 22 and y < 30:
        s = ((y, pos[1] - 3), vel)
    elif y > 29 and y < 33:
        s = ((y, pos[1] - 4), vel)

    return s


def behaviour_policy():
    rand = np.random.uniform()
    if rand < 0.4:
        return 1
    elif rand < 0.55:
        return 5
    elif rand < 0.6:
        return 3
    elif rand < 0.7:
        return 6
    elif rand < 0.9:
        return 7
    else:
        return np.random.choice([0,2,4,8])


def store(Q,C, target_policy):
    #os.system('rm *.pkl')
    f = open('Q.pkl', 'wb')
    pickle.dump(Q, f)
    f.close()
    f = open('C.pkl', 'wb')
    pickle.dump(C, f)
    f.close()
    f = open('pi.pkl', 'wb')
    pickle.dump(target_policy, f)
    f.close()



env = RaceTrack()
n_episodes = int(input())
print(n_episodes)
time.sleep(5)

Q = {}
C = {}
target_policy = {}
curr_row = 0
for n_rows, n_cols in env.state_space:
    for row in range(curr_row, curr_row + n_rows):
        positions = product([row], range(n_cols))
        velocities = product(range(-3,1), range(-2,3))
        states = product(positions, velocities)
        sa_pairs = product(states, range(9))
        for pair in sa_pairs:
            Q[pair] = 0
            C[pair] = 0
    curr_row += n_rows


gamma = 0.95

action_probas = [0.025, 0.4, 0.025, 0.05, 0.025, 0.15, 0.1, 0.2, 0.025]

for i in range(n_episodes):
    if i % 1000 == 0:
        print_episode(i, n_episodes)
    done = False
    obs = env.reset()
    states = [obs]

    a = behaviour_policy()
    actions = [a]
    rewards = []

    while not done:
        obs, reward, done = env.step(a)
        rewards.append(reward)
        states.append(obs)
        a = behaviour_policy()
        actions.append(a)

    G = 0
    W = 1

    for t in range(len(states) - 2, -1, -1):
        G = gamma * G + rewards[t-1]
        s = states[t]
        a = actions[t]
        s = adjust_col(s)

        C[s,a] += W
        Q[s,a] += (W/C[s,a]) * (G - Q[s,a])

        action_values = [Q[s, action] for action in range(9)]
        target_policy[s] = np.argmax(action_values)

        if a == target_policy[s]:
            W *= (1/action_probas[a])
        else:
            break



store(Q,C,target_policy)






'''for row in range(curr_row, curr_row + n_rows):
    Q[row] = [0] * n_cols
curr_row += n_rows

for i,x in Q.items():
    print(i, x)'''
