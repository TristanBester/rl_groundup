# Created by Tristan Bester.
import sys
sys.path.append('../')
import numpy as np
from envs import MountainCar
from functions import LinearPolicy
from semi_gradient_n_step_td import semi_gradient_n_step_td
from utils import TileCoding, print_episode, create_line_plot, \
                  eps_greedy_func_policy, plot_mountain_car_value_function

'''
Episodic Semi-gradient Sarsa used to estimate the optimal state-action
value function for the mountain car environment defined on page 198
of "Reinforcement Learning: An Introduction."
Algorithm available on page 198 of
"Reinforcement Learning: An Introduction."

Book reference:
Sutton, R. and Barto, A., 2014. Reinforcement Learning:
An Introduction. 1st ed. London: The MIT Press.
'''


def semi_gradient_sarsa(env, alpha, gamma, epsilon, n_episodes, tile_coder, action_len):
    # Initialization.
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
                # Update weights.
                q.weights += alpha * np.dot((reward - q.evaluate(x)), x)
            else:
                a_prime = eps_greedy_func_policy(q, obs_prime, epsilon, \
                          tile_coder, env.action_space_size)
                x_prime = tile_coder.get_feature_vector(obs_prime, a_prime)
                # Update weights.
                q.weights += alpha * np.dot((reward + \
                             gamma * q.evaluate(x_prime) - q.evaluate(x)), x)
                obs = obs_prime
                a = a_prime
        # Store steps for plotting.
        all_steps.append(env.steps)
        print_episode(episode, n_episodes)
    # Plot agent performance over training.
    create_line_plot(range(len(all_steps)), all_steps, 'Episode number:', \
    'Number of steps:', 'Number of steps required to reach goal during training:')
    print_episode(n_episodes, n_episodes)
    return q


if __name__ == '__main__':
    n = 4
    gamma = 0.95
    alpha = 0.001
    epsilon = 0.1
    action_vec_dim = 1
    n_episodes_control = 100
    n_episodes_prediction = 1000

    n_tilings = 2
    tile_frac = 1/4
    env = MountainCar()
    pred_env = MountainCar(max_steps=200)
    min_values = [env.min_pos, -env.max_speed]
    max_values = [env.max_pos, env.max_speed]
    tile_coder = TileCoding(min_values, max_values, n_tilings, tile_frac)

    print('Beginning control...\n')
    q = semi_gradient_sarsa(env, alpha, gamma, epsilon, n_episodes_control, \
                            tile_coder, action_vec_dim)

    print('Beginning prediction...\n')
    v = semi_gradient_n_step_td(pred_env, q, n, alpha, gamma, \
                                n_episodes_prediction, tile_coder)

    plot_mountain_car_value_function(env.min_pos, env.max_pos, -env.max_speed, \
                                     env.max_speed, v, tile_coder)
