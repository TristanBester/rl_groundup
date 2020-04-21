import sys
import time
import numpy as np
sys.path.append('../')
from envs import GridWorld
from itertools import product
from utils import print_episode

def eps_greedy_policy(Q, s, epsilon):
    if np.random.uniform() < epsilon:
        return np.random.randint(4)
    else:
        action_values = [Q[s,i] for i in range(4)]
        return np.argmax(action_values)

def policy_proba(policy, s, a, epsilon):
    if policy[s] == a:
        return (epsilon/4) + (1-epsilon)
    else:
        return epsilon/4



def n_step_tree_backup(env, n, alpha, gamma, epsilon, n_episodes):
    sa_pairs = product(range(env.observation_space_size),\
                       range(env.action_space_size))
    Q = dict.fromkeys(sa_pairs, 0.0)
    policy = dict.fromkeys(range(env.observation_space_size), 0)
    states = np.zeros(n)
    actions = np.zeros(n)
    #rewards = np.zeros(n)
    Qs = np.zeros(n)
    deltas = np.zeros(n)
    pis = np.zeros(n)

    decay = lambda x: x-2/n_episodes if x-2/n_episodes > 0.1 else 0.1

    for episode in range(n_episodes):
        done = False
        obs = env.reset()
        action = eps_greedy_policy(Q, obs, epsilon)
        states[0] = obs
        actions[0] = action
        Qs[0] = Q[obs, action]
        t = -1
        tau = -1
        T = np.inf

        while not done or t != T-1:
            t += 1
            #env.render()
            #time.sleep(0.1)
            if t < T:
                obs_prime, reward, done = env.step(action)
                states[(t+1)%n] = obs_prime
                if done:
                    T = t+1
                    deltas[t%n] = reward - Qs[t%n]
                else:
                    deltas[t%n] = reward + gamma * \
                    np.sum([policy_proba(policy, obs_prime, i, epsilon) * \
                    Q[obs_prime, i] for i in range(4)]) - Qs[t%n]
                    action = eps_greedy_policy(Q, obs_prime, epsilon)
                    Qs[(t+1)%n] = Q[obs_prime, action]
                    pis[(t+1)%n] = policy_proba(policy, obs_prime, action, epsilon)
            tau = t-n+1
            if tau > -1:
                Z = 1
                G = Qs[tau%n]
                for k in range(tau,min(tau+n-1, T-1)):
                    G += Z*deltas[k%n]
                    Z *= gamma * Z * pis[(k+1)%n]
                s = states[tau%n]
                a = actions[tau%n]
                Q[s,a] += alpha * (G - Q[s,a])
                action_values = [Q[s,i] for i in range(4)]
                policy[s] = np.argmax(action_values)
        epsilon = decay(epsilon)
        print_episode(episode,n_episodes)
    print_episode(n_episodes, n_episodes)
    return policy

def test_policy(env, policy, n_tests):
    print('Beginning testing...\n')
    time.sleep(2)
    for test in range(n_tests):
        obs = env.reset()
        a = policy[obs]
        env.render()
        done = False
        time.sleep(0.3)
        while not done:
            obs,_,done = env.step(a)
            a = policy[obs]
            env.render()
            time.sleep(0.3)


if __name__ == '__main__':
    n = 4
    alpha = 0.1
    gamma = 1
    epsilon = 1
    n_episodes = 5000
    env = GridWorld()
    policy = n_step_tree_backup(env, n , alpha, gamma, epsilon, n_episodes)
    test_policy(env, policy, 10)
