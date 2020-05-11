# Created by Tristan Bester.
import sys
import numpy as np
sys.path.append('../')
from envs import MountainCar
from functions import LinearPolicy
from utils import TileCoding, print_episode, eps_greedy_func_policy, \
                  create_line_plot

'''
Differential semi-gradient Sarsa used to estimate the optimal
action-value function for the mountain car environment defined
on page 198 of "Reinforcement Learning: An Introduction."
Algorithm available on page 203 of
"Reinforcement Learning: An Introduction."

Book reference:
Sutton, R. and Barto, A., 2014. Reinforcement Learning:
An Introduction. 1st ed. London: The MIT Press.
'''


def differential_semi_gradient_sarsa(env, alpha, beta, epsilon, n_episodes,\
                                     tile_coder, action_vec_dim, stop_threshold):
    # Initialization.
    q = LinearPolicy(tile_coder.total_n_tiles, action_vec_dim, env.action_space_size)
    r_bar = 0
    all_steps = []

    for episode in range(n_episodes):
        done = False
        obs = env.reset()
        a = eps_greedy_func_policy(q, obs, epsilon, tile_coder, \
                                   env.action_space_size)

        while not done:
            obs_prime, reward, done = env.step(a)
            a_prime = eps_greedy_func_policy(q, obs_prime, epsilon, tile_coder,\
                                       env.action_space_size)
            x = tile_coder.get_feature_vector(obs, a)
            x_prime = tile_coder.get_feature_vector(obs_prime, a_prime)
            delta = reward - r_bar + q.evaluate(x_prime) - q.evaluate(x)
            r_bar += beta * delta
            # Update weights.
            q.weights += alpha * delta * x
            obs = obs_prime
            a = a_prime
        # Stop training if state-action value function has converged.
        if len(all_steps) > 10 and sum(all_steps[-10:]) < stop_threshold:
            break
        # Store steps for plotting.
        all_steps.append(env.steps)
        print_episode(episode, n_episodes)
    # Plot agent performance during training.
    create_line_plot(range(len(all_steps)), all_steps, 'Episode number:', \
    'Number of steps:', 'Number of steps required to reach goal during training:')
    print_episode(n_episodes, n_episodes)
    return q


if __name__ == '__main__':
    beta = 0.9
    gamma = 0.9
    alpha = 0.01
    epsilon = 0.1
    n_tilings = 2
    tile_frac = 1/4
    n_episodes = 100
    action_vec_dim = 1
    stop_threshold = 1000
    env = MountainCar()
    min_values = [env.min_pos, -env.max_speed]
    max_values = [env.max_pos, env.max_speed]
    tile_coder = TileCoding(min_values, max_values, n_tilings, tile_frac)
    q = differential_semi_gradient_sarsa(env, alpha, beta, epsilon, n_episodes, \
                                         tile_coder, action_vec_dim, stop_threshold)
