import sys
import numpy as np
sys.path.append('../')
from itertools import product
from envs import WindyGridWorld
from n_step_td_prediction import n_step_td_pred
from td_zero_prediction import td_pred
from utils import print_episode, create_value_func_plot, print_grid_world_actions

def eps_greedy_policy(Q, s, epsilon):
    if np.random.uniform() < epsilon:
        return np.random.randint(4)
    else:
        action_values = [Q[s,i] for i in range(4)]
        return np.argmax(action_values)


def n_step_sarsa(env, n, alpha, gamma, epsilon, n_episodes):
    sa_pairs = product(range(env.observation_space_size), \
                       range(env.action_space_size))
    Q = dict.fromkeys(sa_pairs, 0.0)
    states = np.zeros(n)
    actions =np.zeros(n)
    rewards = np.zeros(n)
    epsilon_start = epsilon

    #decay = lambda x: x - 2/n_episodes if x - 2/n_episodes > 1e-5 else 1e-5
    # eps will hit baseline after 10% of the total episodes have finished.
    decay = lambda x: x - (10/n_episodes)*epsilon_start if x - (10/n_episodes)*epsilon_start > 0 else 1e-5
    #decay = lambda i,x: 1/(0.0005 *(i+1))
    #decay = lambda i,x: x/(i+1) if x/(i+1) > 0.001 else 0.001

    for episode in range(n_episodes):
        done = False
        obs = env.reset()
        t = 0
        T = np.inf
        states[t] = obs
        a = eps_greedy_policy(Q, obs, epsilon)
        actions[t] = a

        step_count = 0
        while not done or tau != T-1:
            step_count += 1
            if t < T:
                obs_prime, reward, done = env.step(a)
                rewards[(t+1)%n] = reward
                states[(t+1)%n] = obs_prime
                if done:
                    T = t+1
                else:
                    a = eps_greedy_policy(Q, obs_prime, epsilon)
                    actions[(t+1)%n] = a
            tau = t - n + 1
            if tau > -1:
                G = np.sum([gamma ** (i-tau-1) * rewards[i%n] for i in range(tau + 1, min(tau+n, T))])
                if tau + n < T:
                    state = states[(tau+n)%n]
                    action = actions[(tau+n)%n]
                    G += gamma ** n * Q[state, action]
                s = states[tau%n]
                a = actions[tau%n]
                Q[s,a] += alpha * (G - Q[s,a])
            t += 1
        epsilon = decay(epsilon)
        if episode % 1 == 0:
            print_episode(episode, n_episodes)
            print(f'Step counter: {step_count}, Eps: {epsilon}')
    print_episode(n_episodes, n_episodes)
    return Q


def create_greedy_policy(env, Q):
    policy = {}
    for s in range(env.observation_space_size):
        action_values = [Q[s,i] for i in range(env.action_space_size)]
        policy[s] = np.argmax(action_values)
    return policy


if __name__ == '__main__':
    n = 25
    alpha = 0.4
    gamma = 0.7
    epsilon = 1.0
    n_episodes = 1000
    env = WindyGridWorld()
    Q = n_step_sarsa(env, n, alpha, gamma, epsilon, n_episodes)
    policy = create_greedy_policy(env, Q)



    print_grid_world_actions(np.array(list(policy.values())), (7, 10))
    #V = n_step_td_pred(env, policy, n, alpha, gamma, 1000)
    V =  td_pred(env, policy, alpha, gamma, 100000)
    create_value_func_plot(V, (7, 10), 'Windy grid world value function:')
