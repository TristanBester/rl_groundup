from policy_iteration import evaluate_policy
import numpy as np
import gym

def value_iteration(env, gamma, epsilon):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    V = np.zeros(n_states)
    policy = np.zeros(n_states)
    max_delta = np.inf

    while max_delta > epsilon:
        max_delta = -1
        for state in range(n_states):
            action_values = []
            for a in range(n_actions):
                action_value = 0
                for transition_proba, n_state, reward, done in env.P[state][a]:
                    action_value += transition_proba * (reward + gamma * V[n_state])
                action_values.append(action_value)
            last = V[state]
            V[state] = np.max(action_values)
            # Store the greedy action w.r.t. V(s) so we don't need to
            # recalute action values at end of value iteration.
            policy[state] = np.argmax(action_values)
            delta = abs(V[state] - last)
            if delta > max_delta:
                max_delta =  delta

    return V, policy

if __name__ ==  '__main__':
    env = gym.make('FrozenLake-v0')
    gamma = 1.0
    epsilon = 1e-3
    V, policy = value_iteration(env, gamma, epsilon)
    evaluate_policy(env, policy, 1000)
    env.close()
