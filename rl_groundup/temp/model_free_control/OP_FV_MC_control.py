import sys
import numpy as np
import gym
sys.path.append('../')
from utils import create_surface_plot


env = gym.make('Blackjack-v0')
n_hands = env.observation_space[0].n
n_dealer = env.observation_space[1].n
n_usable = env.observation_space[2].n
n_actions = env.action_space.n

Q = {}
policy = {}
returns = {}

for i in range(n_hands):
    for j in range(n_dealer):
        for k in range(n_usable):
            for a in range(n_actions):
                state = (i,j,k)
                Q[state, a] = 0
                policy[state] = 0
                returns[state, a] = []

#print(Q)

n_episodes = 10000
epsilon = 0.2
for episode in range(n_episodes):
    done = False
    sa_pairs = []
    fin_reward = 0
    obs = env.reset()

    while not done:
        action = policy[obs]
        if (obs, action) not in sa_pairs:
            sa_pairs.append((obs,action))
        obs, reward, done, info = env.step(action)
        fin_reward = reward

    for s,a in sa_pairs:
        G = fin_reward
        returns[s,a].append(G)
        Q[s,a] = np.mean(returns[s,a])

    for s,_ in sa_pairs:
        action_values = [Q[s,i] for i in range(2)]
        a_opt = np.argmax(action_values)
        if np.random.random() < epsilon:
            policy[s] = env.action_space.sample()
        else:
            policy[s] = a_opt



keys = np.array(list(Q.keys()))
states = np.array(keys[:, 0]).reshape((-1,1))
values = np.array(list(Q.values()))
x_u = []
y_u = []
z_u = []
for state,value in zip(states, values):
    if state[0][2]:
        x_u.append(state[0][0])
        y_u.append(state[0][1])
        z_u.append(value)

create_surface_plot(x_u,y_u,z_u, '','','')










#
