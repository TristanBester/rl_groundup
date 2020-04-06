import policy_evaluation
import numpy as np
import gym

def create_greedy_policy(V, env, gamma):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    greedy_policy = np.zeros(n_states)

    for state in range(16):
        action_values = []
        for a in range(n_actions):
            action_value = 0
            for transition_proba, n_state, reward, done in env.P[state][a]:
                action_value += transition_proba * (reward + gamma * V[n_state])
            action_values.append(action_value)
        greedy_policy[state] = np.argmax(action_values)
    return greedy_policy


def usable_greedy_policy(greedy_policy):
    policy = lambda s,a: 1 if greedy_policy[s] == a else  0
    return policy


def policy_iteration(env, epsilon, gamma):
    # Arbitray initial policy.
    policy = policy_evaluation.stochastic_policy

    V = policy_evaluation.evalute_policy(policy, env, gamma, epsilon)
    greedy_policy = create_greedy_policy(V, env, gamma)
    last_policy = np.zeros(greedy_policy.shape)

    while not np.all(greedy_policy == last_policy):
        V = policy_evaluation.evalute_policy(policy, env, gamma, epsilon)
        greedy_policy = create_greedy_policy(V, env, gamma)
        policy = usable_greedy_policy(greedy_policy)
        last_policy = greedy_policy.copy()

    return last_policy


def evaluate_policy(env, policy, num_games):
    scores =0
    for i in range(num_games):
        obs = env.reset()
        done = False
        r = -1
        a = int(policy[int(obs)])
        while not done:
            obs, reward, done, info = env.step(a)
            r = reward
            a = int(policy[int(obs)])
        scores += r
    scores = round((scores/num_games) * 100, 2)
    print(f'Percentage of games won: {scores}%')


if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    epsilon = 1e-5
    gamma = 1.0
    policy = policy_iteration(env, epsilon, gamma)
    evaluate_policy(env, policy, 1000)
    env.close()
