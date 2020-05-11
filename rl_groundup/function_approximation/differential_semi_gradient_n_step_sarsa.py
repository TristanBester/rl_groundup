# Created by Tristan Bester.
import sys
import numpy as np
sys.path.append('../')
from envs import MountainCar
from functions import LinearPolicy
from utils import print_episode, TileCoding, eps_greedy_func_policy, \
                  create_line_plot

'''
Differential semi-gradient n-step Sarsa used to estimate the optimal
action-value function for the mountain car environment defined
on page 198 of "Reinforcement Learning: An Introduction."
Algorithm available on page 206 of
"Reinforcement Learning: An Introduction."

Book reference:
Sutton, R. and Barto, A., 2014. Reinforcement Learning:
An Introduction. 1st ed. London: The MIT Press.
'''


def differential_semi_gradient_n_step_sarsa(env, n, alpha, beta, epsilon, \
                        n_episodes, tile_coder, action_vec_dim, stop_threshold):
    # Initialization.
    q = LinearPolicy(tile_coder.total_n_tiles, action_vec_dim, env.action_space_size)
    r_bar = 0
    states = [None] * n
    actions = np.zeros(n)
    rewards = np.zeros(n)
    all_steps = []

    for episode in range(n_episodes):
        done = False
        obs = env.reset()
        states[0] = obs
        a = eps_greedy_func_policy(q, obs, epsilon, tile_coder, \
                                   env.action_space_size)
        t = 0
        tau = -1

        while not done:
            obs, reward, done = env.step(a)
            states[(t+1)%n] = obs
            rewards[(t+1)%n] = reward
            a = eps_greedy_func_policy(q, obs, epsilon, tile_coder, \
                                       env.action_space_size)
            actions[(t+1)%n] = a
            tau = t-n+1
            if tau > -1:
                x = tile_coder.get_feature_vector(states[tau%n], actions[tau%n])
                x_n = tile_coder.get_feature_vector(states[(tau+n)%n], \
                                                    actions[(tau+n)%n])
                summ = np.sum([rewards[i%n] - r_bar for i in range(tau+1, tau+n)])
                delta =  summ + q.evaluate(x_n) - q.evaluate(x)
                r_bar += beta * delta
                q.weights += alpha * delta * x
            t += 1
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
    n = 4
    beta = 0.5
    alpha = 0.5
    epsilon = 0.1
    n_tilings = 2
    tile_frac = 1/4
    n_episodes = 50
    action_vec_dim = 1
    stop_threshold = 1000

    env = MountainCar()
    min_values = [env.min_pos, -env.max_speed]
    max_values = [env.max_pos, env.max_speed]

    print(alpha, beta, '\n')
    tile_coder = TileCoding(min_values, max_values, n_tilings, tile_frac)
    q = differential_semi_gradient_n_step_sarsa(env, n, alpha, beta, epsilon, \
                        n_episodes, tile_coder, action_vec_dim, stop_threshold)
