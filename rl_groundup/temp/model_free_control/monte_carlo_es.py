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

n_episodes = 50000
for episode in range(n_episodes):
    if episode % 10000 == 0:
        print(episode)
    done = False


    # This returns S0 with all possible states having a non_zero probability
    obs = env.reset()
    # The returns an action such that all actions have a non-zero proba of being selected
    action = env.action_space.sample()
    episode_sa = [(obs,action)]
    
    obs, reward, done, info = env.step(action)
    episode_reward = 0

    while not done:
        action = policy[obs]
        sa_pair = (obs, action)
        obs, reward, done, info = env.step(action)
        if sa_pair not in episode_sa:
            episode_sa.append(sa_pair)
        episode_reward = reward

    for s,a in episode_sa:
        G = episode_reward
        returns[s,a].append(G)
        Q[s,a] = np.mean(returns[s,a])

    for s,_ in episode_sa:
        action_values = [Q[s,a] for a in range(env.action_space.n)]
        policy[s] = np.argmax(action_values)



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
