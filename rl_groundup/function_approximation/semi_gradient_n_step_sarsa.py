# Created by Tristan Bester.
import sys
sys.path.append('../')
import numpy as np
from envs import MountainCar
from functions import LinearPolicy
from utils import TileCoding, eps_greedy_func_policy, create_line_plot, \
                  print_episode

'''
Episodic semi-gradient n-step Sarsa used to estimate the optimal
action-value function for the mountain car environment defined
on page 198 of "Reinforcement Learning: An Introduction."
Algorithm available on page 200 of
"Reinforcement Learning: An Introduction."

Book reference:
Sutton, R. and Barto, A., 2014. Reinforcement Learning:
An Introduction. 1st ed. London: The MIT Press.
'''


def semi_gradient_n_step_sarsa(env, n, alpha, gamma, epsilon, n_episodes, \
                               tile_coder, action_len, stop_threshold):
    # Initialization.
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
                # Calculate n-step return.
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
                # Update weights.
                q.weights += alpha * (np.dot((G - q.evaluate(x)),x))
            t += 1
        print_episode(episode, n_episodes)
        # Stop training if state-action value function has converged.
        if len(all_steps) > 10 and sum(all_steps[-10:]) < stop_threshold:
            break
        # Store steps for plotting.
        all_steps.append(env.steps)
    # Plot agent performance during training.
    create_line_plot(range(len(all_steps)), all_steps, 'Episode number:', \
    'Number of steps:', 'Number of steps required to reach goal during training:')
    print_episode(n_episodes, n_episodes)
    return q


if __name__ == '__main__':
    n = 3
    gamma = 0.8
    alpha = 0.005
    epsilon = 0.1
    n_tilings = 2
    tile_frac = 1/4
    action_len = 1
    n_episodes = 100
    stop_threshold = 1000
    env = MountainCar()
    min_values = [env.min_pos, -env.max_speed]
    max_values = [env.max_pos, env.max_speed]
    tile_coder = TileCoding(min_values, max_values, n_tilings, tile_frac)
    q = semi_gradient_n_step_sarsa(env, n, alpha, gamma, epsilon, \
        n_episodes, tile_coder, action_len, stop_threshold)
