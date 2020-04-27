import sys
import time
import heapq
import numpy as np
sys.path.append('../')
from envs import Maze
from utils import print_episode
from itertools import product, tee


def eps_greedy_policy(Q, s, epsilon):
    if np.random.uniform() < epsilon:
        return np.random.randint(4)
    else:
        action_values = [Q[s, i] for i in range(4)]
        return np.argmax(action_values)


def prioritized_sweeping(env, alpha, gamma, epsilon, theta, n_episodes):
    sa_pairs = product(range(env.observation_space_size), \
                       range(env.action_space_size))
    it_one, it_two = tee(sa_pairs)
    Q = dict.fromkeys(it_one, 0)
    model = {pair:(0,0) for pair in it_two}

    for episode in range(n_episodes):
        done = False
        obs = env.reset()
        action = eps_greedy_policy(Q, obs, epsilon)
        q = []

        while not done:
            obs_prime, reward, done = env.step(action)
            model[obs, action] = (reward, obs_prime)
            opt_a = np.argmax([Q[obs_prime, i] for i in range(4)])
            P = abs(reward + gamma * Q[obs_prime, opt_a] - Q[obs, action])
            if P > theta:
                # Negative P used to allow a min binary heap to be used.
                q.append((-P, (obs, action)))
            obs = obs_prime
            action = eps_greedy_policy(Q, obs, epsilon)

        heapq.heapify(q)
        counter = 0
        while len(q) > 0 and counter < n:
            counter += 1
            _, (s,a) = heapq.heappop(q)
            r, s_prime = model[s,a]
            opt_a = np.argmax([Q[s_prime, i] for i in range(4)])
            Q[s, a] += alpha * (reward + gamma * Q[s_prime, opt_a] - Q[s,a])

            for s_, a_ in env.get_predecessor_states(s):
                r_, _ = model[s_, a_]
                opt_a = np.argmax([Q[s, i] for i in range(4)])
                P = abs(r_ + gamma * Q[s, opt_a] - Q[s_, a_])
                if P > theta:
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


def create_greedy_policy(env, Q):
    policy = {}
    for s in range(env.observation_space_size):
        action_values = [Q[s, i] for i in range(env.action_space_size)]
        policy[s] = np.argmax(action_values)
    return policy


def test_policy(env, policy, n_tests):
    input('Press any key to begin tests.')
    for i in range(n_tests):
        done = False
        obs = env.reset()
        env.render()
        time.sleep(0.3)
        while not done:
            a = policy[obs]
            obs, _, done = env.step(a)
            env.render()
            time.sleep(0.3)


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
    policy = create_greedy_policy(env, Q)
    test_policy(env, policy, n_tests)
