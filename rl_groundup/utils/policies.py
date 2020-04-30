# Created by Tristan Bester.
import time
import numpy as np


def eps_greedy_policy(Q, s, epsilon, n_actions):
    '''An epsilon-greedy policy.'''
    if np.random.uniform() < epsilon:
        return np.random.randint(n_actions)
    else:
        action_values = [Q[s, i] for i in range(n_actions)]
        return np.argmax(action_values)


def create_greedy_policy(Q, n_states, n_actions):
    '''Creates a policy that is greedy w.r.t. the given state-action
    value function.'''
    policy = {}
    for s in range(n_states):
        action_values = [Q[s, i] for i in range(n_actions)]
        policy[s] = np.argmax(action_values)
    return policy


def test_policy(env, policy, n_tests):
    '''Tests the given policy in the specified environment.'''
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
