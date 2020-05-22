import sys
import numpy as np
sys.path.append('../')
from functions import ExponentialSoftmax
from envs import GridWorld
from utils import encode_sa_pair, print_episode
import matplotlib.pyplot as plt


def REINFORCE(env, alpha, gamma, n_episodes):
    policy = ExponentialSoftmax(env.observation_space_size * env.action_space_size)

    for episode in range(n_episodes):
        done = False
        obs = env.reset()
        all_sa_pairs = [encode_sa_pair(obs, a, env.observation_space_size, \
        env.action_space_size) for a in range(env.action_space_size)]
        a = policy.sample_action(all_sa_pairs)
        states = [obs]
        actions = [a]
        rewards = [None]

        while not done:
            obs, reward, done = env.step(a)
            all_sa_pairs = [encode_sa_pair(obs, a, env.observation_space_size, \
            env.action_space_size) for a in range(env.action_space_size)]
            a = policy.sample_action(all_sa_pairs)
            states.append(obs)
            actions.append(a)
            rewards.append(reward)

        for t in range(len(states)):
            G_t = sum(rewards[t+1:])
            all_sa_pairs = [encode_sa_pair(states[t], a, env.observation_space_size, \
            env.action_space_size) for a in range(env.action_space_size)]
            policy.weights += alpha * (gamma ** t) * G_t * \
                              policy.eligibility_vector(actions[t], all_sa_pairs)

        if episode % 100 == 0:
            print_episode(episode, n_episodes)
    print_episode(n_episodes, n_episodes)
    return policy


def test_policy(env, policy, n_tests):
    # MOve this function.
    import time
    input('Press any key to begin tests.')
    for i in range(n_tests):
        done = False
        obs = env.reset()
        env.render()
        time.sleep(0.3)
        while not done:
            all_sa_pairs = [encode_sa_pair(obs, a, env.observation_space_size, \
            env.action_space_size) for a in range(env.action_space_size)]
            a = policy.greedy_action(all_sa_pairs)
            obs, _, done = env.step(a)
            env.render()
            time.sleep(0.3)


if __name__ == '__main__':
    gamma = 1
    n_tests = 10
    n_episodes = 20000
    alpha = 0.00000001
    env = GridWorld()
    policy = REINFORCE(env, alpha, gamma, n_episodes)
    test_policy(env, policy, n_tests)
