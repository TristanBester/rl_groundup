import sys
sys.path.append('../')
import numpy as np
from envs import MountainCar
from functions import LinearPolicy
from utils import TileCoding, eps_greedy_func_policy, create_line_plot,\
                  print_episode


def semi_gradient_n_step_sarsa(env, n, alpha, gamma, epsilon,\
                               n_episodes, tile_coder, action_len, stop_threshold):
    q = LinearPolicy(tile_coder.total_n_tiles, action_len, env.action_space_size)
    states = [None] * n
    actions = np.zeros(n)
    rewards = np.zeros(n)
    all_steps = []

    for episode in range(n_episodes):
        done = False
        obs = env.reset()
        states[0] = obs
        a = eps_greedy_func_policy(q, obs, epsilon, tile_coder,\
                                   env.action_space_size)
        t = 0
        tau = -1
        T = np.inf

        while not done or tau != T-1:
            if t < T:
                obs_prime, reward, done = env.step(a)
                rewards[(t+1)%n] = reward
                states[(t+1)%n] = obs_prime
                if done:
                    T = t+1
                else:
                    a = eps_greedy_func_policy(q, obs_prime, epsilon, \
                        tile_coder, env.action_space_size)
                    actions[(t+1)%n] = a
            tau = t-n+1
            if tau > -1:
                G = np.sum([gamma**(i-tau-1)*rewards[i%n] \
                    for i in range(tau+1, min(tau+n,T))])
                if tau + n < T:
                    s = states[(tau+n)%n]
                    a = actions[(tau+n)%n]
                    x = tile_coder.get_feature_vector(s, a)
                    G += gamma**n * q.evaluate(x)
                s = states[tau%n]
                a = actions[tau%n]
                x = tile_coder.get_feature_vector(s, a)
                q.weights += alpha * (np.dot((G - q.evaluate(x)),x))
            t += 1
        print_episode(episode, n_episodes)
        # Test if policy has converged.
        if len(all_steps) > 10 and sum(all_steps[-10:]) < stop_threshold:
            break
        all_steps.append(env.steps)
    create_line_plot(range(len(all_steps)), all_steps, 'Episode number:', \
    'Number of steps:', 'Number of steps required to reach goal during training:')
    print_episode(n_episodes, n_episodes)
    return q



if __name__ == '__main__':
    n = 3
    env = MountainCar()
    alpha = 0.005
    gamma = 0.8
    n_episodes = 100
    epsilon = 0.1
    stop_threshold = 1000
    min_values = [env.min_pos, -env.max_speed]
    max_values = [env.max_pos, env.max_speed]
    n_tilings = 2
    tile_frac = 1/4
    tile_coder = TileCoding(min_values, max_values, n_tilings, tile_frac)
    q = semi_gradient_n_step_sarsa(env, n, alpha, gamma, epsilon, \
        n_episodes, tile_coder, 1, stop_threshold)
