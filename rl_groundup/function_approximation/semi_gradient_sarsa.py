import sys
sys.path.append('../')
import numpy as np
from envs import MountainCar
from functions import LinearPolicy
from semi_gradient_td_zero import semi_gradient_td_zero
from utils import TileCoding, print_episode, create_line_plot, \
                  eps_greedy_func_policy, plot_mountain_car_value_function


def semi_gradient_sarsa(env, alpha, gamma, epsilon, n_episodes, tile_coder, action_len):
    q = LinearPolicy(tile_coder.total_n_tiles, action_len, env.action_space_size)
    all_steps = []

    for episode in range(n_episodes):
        done = False
        obs = env.reset()
        a = eps_greedy_func_policy(q, obs, epsilon, tile_coder, \
                                   env.action_space_size)

        while not done:
            obs_prime, reward, done = env.step(a)
            x = tile_coder.get_feature_vector(obs, a)
            if done:
                q.weights += alpha * np.dot((reward - q.evaluate(x)), x)
            else:
                a_prime = eps_greedy_func_policy(q, obs_prime, epsilon, \
                          tile_coder, env.action_space_size)
                x_prime = tile_coder.get_feature_vector(obs_prime, a_prime)
                q.weights += alpha * np.dot((reward + \
                             gamma * q.evaluate(x_prime) - q.evaluate(x)), x)
                obs = obs_prime
                a = a_prime
        all_steps.append(env.steps)
        print_episode(episode, n_episodes)
    create_line_plot(range(len(all_steps)), all_steps, 'Episode number:', \
    'Number of steps:', 'Number of steps required to reach goal during training:')
    print_episode(n_episodes, n_episodes)
    return q


if __name__ == '__main__':
    env = MountainCar()
    alpha = 0.001
    gamma = 0.95
    n_episodes = 100
    epsilon = 0.1
    min_values = [env.min_pos, -env.max_speed]
    max_values = [env.max_pos, env.max_speed]
    n_tilings = 2
    tile_frac = 1/4
    tile_coder = TileCoding(min_values, max_values, n_tilings, tile_frac)
    print('Beginning control...')
    q = semi_gradient_sarsa(env, alpha, gamma, epsilon, n_episodes, tile_coder, 1)
    print('Beginning prediction...')
    env = MountainCar(max_steps = 100)
    v = semi_gradient_td_zero(env, q, alpha, gamma, 10, tile_coder)
    plot_mountain_car_value_function(env.min_pos, env.max_pos, -env.max_speed, \
                                     env.max_speed, v, tile_coder)
