# Created by Tristan Bester.
import sys
import heapq
import numpy as np
sys.path.append('../')
from envs import Maze
from itertools import product, tee
from utils import print_episode, eps_greedy_policy, create_greedy_policy, \
                  test_policy

'''
Prioritized sweeping used to find an optimal policy for the maze environment
described on page 135 of "Reinforcement Learning: An Introduction."
Algorithm available on page 140.

Book reference:
Sutton, R. and Barto, A., 2014. Reinforcement Learning:
An Introduction. 1st ed. London: The MIT Press.
'''


def prioritized_sweeping(env, alpha, gamma, epsilon, theta, n_episodes):
    # Create iterators.
    sa_pairs = product(range(env.observation_space_size), \
                       range(env.action_space_size))
    it_one, it_two = tee(sa_pairs)

    # Initialize state-action value function and model.
    Q = dict.fromkeys(it_one, 0)
    model = {pair:(0,0) for pair in it_two}

    for episode in range(n_episodes):
        done = False
        obs = env.reset()
        action = eps_greedy_policy(Q, obs, epsilon, env.action_space_size)
        q = []

        while not done:
            obs_prime, reward, done = env.step(action)
            model[obs, action] = (reward, obs_prime)
            opt_a = np.argmax([Q[obs_prime, i] for i in range(4)])
            P = abs(reward + gamma * Q[obs_prime, opt_a] - Q[obs, action])
            # Maintain priority queue of each state-action pair whose estimated
            # value changes nontrivially. Prioritized by size of change.
            if P > theta:
                # Negative P used to allow a min binary heap to be used.
                q.append((-P, (obs, action)))
            obs = obs_prime
            action = eps_greedy_policy(Q, obs, epsilon, env.action_space_size)

        counter = 0
        heapq.heapify(q)
        while len(q) > 0 and counter < n:
            counter += 1
            _, (s,a) = heapq.heappop(q)
            r, s_prime = model[s,a]
            opt_a = np.argmax([Q[s_prime, i] for i in range(4)])
            Q[s, a] += alpha * (reward + gamma * Q[s_prime, opt_a] - Q[s,a])

            # Determine the effect the change of value has on predecessor state-
            # action pairs' values.
            for s_, a_ in env.get_predecessor_states(s):
                r_, _ = model[s_, a_]
                opt_a = np.argmax([Q[s, i] for i in range(4)])
                P = abs(r_ + gamma * Q[s, opt_a] - Q[s_, a_])

                # Add predecessor state-action pairs to priority queue if change
                # causes their value to change nontrivially.
                if P > theta:
                    # If state-action pair already in queue, keep only the
                    # higher priority entry.
                    ls = [i for i in q if i[1] == (s_,a_)]
                    if len(ls) > 0:
                        if ls[0][0] > -P:
                            q.remove(ls[0])
                            heapq.heapify(q)
                            heapq.heappush(q, ((-P, (s_, a_))))
                    else:
                        heapq.heappush(q, ((-P, (s_, a_))))

        print_episode(episode, n_episodes)
    print_episode(n_episodes, n_episodes)
    return Q


if __name__ == '__main__':
    n = 10
    alpha = 0.1
    gamma = 0.95
    epsilon = 0.1
    theta = 0.1
    n_episodes = 300
    n_tests = 10
    env = Maze()
    Q = prioritized_sweeping(env, alpha, gamma, epsilon, theta, n_episodes)
    policy = create_greedy_policy(Q, env.observation_space_size, \
                                  env.action_space_size)
    test_policy(env, policy, n_tests)
