import sys
import numpy as np
sys.path.append('../')
from envs import GridWorld
from utils import print_episode, encode_state, encode_sa_pair
from functions import ExponentialSoftmax, LinearValueFunction

def actor_critic_eligibility_traces(env, eta, alpha_th, alpha_w, lambda_th, lambda_w, \
                                    gamma, n_episodes):

    policy = ExponentialSoftmax(env.observation_space_size * env.action_space_size)
    v = LinearValueFunction(env.observation_space_size)
    z_th = np.zeros(env.observation_space_size * env.action_space_size)
    z_w = np.zeros (env.observation_space_size)
    R_bar = 0

    for episode in range(n_episodes):
        done = False
        obs = env.reset()
        obs_vec = encode_state(obs, env.observation_space_size)

        while not done:
            sa_pairs = [encode_sa_pair(obs, a, env.observation_space_size, \
                        env.action_space_size) for a in range(env.action_space_size)]
            a =  policy.sample_action(sa_pairs)
            obs_prime, reward, done = env.step(a)
            obs_prime_vec = encode_state(obs_prime, env.observation_space_size)
            delta = reward - R_bar + v.evaluate(obs_prime_vec) - v.evaluate(obs_vec)
            R_bar += eta * delta
            z_w = lambda_w * z_w + obs_vec
            z_th = lambda_th * z_th + policy.eligibility_vector(a, sa_pairs)
            v.weights += alpha_w * delta * z_w
            policy.weights += alpha_th * delta * z_th
            obs_vec =  obs_prime_vec
            obs = obs_prime

        print_episode(episode, n_episodes)
    print_episode(n_episodes, n_episodes)
    return policy

def test_policy(env, policy, n_tests):
    # Move this function.
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
    eta = 1e-8
    gamma = 1
    alpha_w = 0.01
    alpha_th = 1e-6
    lambda_w = 0.5
    lambda_th = 0.5
    n_episodes = 12000
    n_tests = 10
    env = GridWorld()
    policy = actor_critic_eligibility_traces(env, eta, alpha_th, alpha_w, \
             lambda_th, lambda_th, gamma, n_episodes)
    test_policy(env, policy, n_tests)
