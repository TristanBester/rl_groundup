import sys
import numpy as np
sys.path.append('../')
from envs import GridWorld
from utils import print_episode, encode_state, encode_sa_pair
from functions import ExponentialSoftmax, LinearValueFunction


def one_step_actor_critic(env, alpha_th, alpha_w, gamma, n_episodes):
    policy = ExponentialSoftmax(env.observation_space_size*env.action_space_size)
    v = LinearValueFunction(env.observation_space_size)

    for episode in range(n_episodes):
        done = False
        obs = env.reset()
        obs_vec = encode_state(obs, env.observation_space_size)
        I = 1

        while not done:
            sa_pairs = [encode_sa_pair(obs, a, env.observation_space_size, \
                       env.action_space_size) for a in range(env.action_space_size)]
            a = policy.sample_action(sa_pairs)
            obs_prime, reward, done = env.step(a)
            obs_prime_vec = encode_state(obs_prime, env.observation_space_size)
            delta = reward + gamma * v.evaluate(obs_prime_vec) - v.evaluate(obs_vec)
            v.weights += alpha_w * I  * delta * obs_vec
            policy.weights += alpha_th  * I * delta * policy.eligibility_vector(a,
            sa_pairs)
            I *= I
            obs_vec = obs_prime_vec
            obs = obs_prime

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
    alpha_w = 0.01
    alpha_th = 1e-7
    n_episodes = 5000
    env = GridWorld()
    policy = one_step_actor_critic(env, alpha_th, alpha_w, gamma, n_episodes)
    test_policy(env, policy, n_tests)
