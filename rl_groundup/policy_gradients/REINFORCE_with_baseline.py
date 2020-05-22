import sys
import numpy as np
sys.path.append('../')
from envs import ShortCorridor
from functions import ExponentialSoftmax, LinearValueFunction
from utils import print_episode, encode_sa_pair, encode_state, create_line_plot


def REINFORCE_baseline(env, alpha_th, alpha_w, gamma, n_episodes):
    policy = ExponentialSoftmax(env.observation_space_size * env.action_space_size)
    v = LinearValueFunction(env.observation_space_size)

    returns = []
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
            x_t = encode_state(states[t], env.observation_space_size)
            delta = G_t - v.evaluate(x_t)
            v.weights += alpha_w * (gamma ** t) * delta * x_t
            all_sa_pairs = [encode_sa_pair(states[t], a, env.observation_space_size, \
            env.action_space_size) for a in range(env.action_space_size)]
            policy.weights += alpha_th * (gamma ** t) * G_t * delta * \
                              policy.eligibility_vector(actions[t], all_sa_pairs)

        returns.append(sum(rewards[1:]))
        print_episode(episode, n_episodes)
    print_episode(n_episodes, n_episodes)
    return (policy, np.array(returns))


if __name__ == '__main__':
    gamma = 1
    alpha_w = 0.001
    alpha_th = 0.000001
    n_episodes = 1000
    env = ShortCorridor()

    all_returns = np.array([REINFORCE_baseline(env, alpha_th, alpha_w, gamma, \
                  n_episodes)[1] for i in range(150)])
    all_returns = np.sum(all_returns, axis=0)
    all_returns = all_returns / all_returns.shape[0]
    create_line_plot(range(all_returns.shape[0]), all_returns, 'Episode number:', \
    'Average return:', 'Returns averaged over 150 independent runs:')
