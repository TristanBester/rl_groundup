# Created by Tristan Bester.
import time
import numpy as np
from .encoding import encode_sa_pair


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


def eps_greedy_func_policy(q, state, epsilon, tile_coder, n_actions):
    '''Epsilon-greedy policy. Implementation specific to use with linear
    function approximation and tile coding.'''
    if np.random.uniform() < epsilon:
        return np.random.randint(n_actions)
    else:
        feature_vectors = tile_coder.get_feature_vectors_for_actions(state,\
                          n_actions)
        return q.greedy_action(feature_vectors)


def eps_greedy_policy_bin_features(q, state, epsilon, n_states, n_actions):
    '''Epsilon-greedy policy. Implementation specific to use with linear
    function approximation and binary features.'''
    if np.random.uniform() < epsilon:
        return np.random.randint(n_actions)
    else:
        encoded_vectors = [encode_sa_pair(state, a, n_states, n_actions) for a \
                           in range(n_actions)]
        return q.greedy_action(encoded_vectors)


def test_linear_policy(env, policy, n_tests):
    '''Tests the given policy in the specified environment. Implementation
    specific to use with linear function approximation and binary features.'''
    input('Press any key to begin tests.')
    for i in range(n_tests):
        done = False
        obs = env.reset()
        env.render()
        time.sleep(0.3)
        while not done:
            feature_vectors = [encode_sa_pair(obs, a, env.observation_space_size,\
                    env.action_space_size) for a in range(env.action_space_size)]
            a = policy.greedy_action(feature_vectors)
            obs, _, done = env.step(a)
            env.render()
            time.sleep(0.3)
