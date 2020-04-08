import matplotlib.pyplot as plt
import numpy as np
import gym

def stochastic_policy(state, action):
    '''Uniform random policy.'''
    return 0.25

def evalute_policy(policy, env, gamma ,epsilon):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    V = np.zeros(n_states)
    V_last = np.ones(n_states)

    while(abs(np.max(V - V_last)) > epsilon):
        V_last = V.copy()
        for state in range(n_states-1):
            n_value = 0
            for a in range(n_actions):
                for transition_proba, n_state, reward, done in env.P[state][a]:
                    state_action_proba = policy(state,a)
                    n_state_val = V[n_state]
                    n_value += state_action_proba * (reward + gamma * transition_proba * n_state_val)
            V[state] = n_value
    return V


def show_policy(V, shape):
    V = np.reshape(V, shape)

    plt.matshow(V.reshape(4,4), cmap='cool')
    plt.title('State-value function of uniform random policy:')
    plt.colorbar()
    fig = plt.gcf()
    fig.set_size_inches((7,7))
    plt.show()



if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    epsilon = 1e-5
    gamma = 1.0
    V = evalute_policy(stochastic_policy, env, gamma, epsilon)
    show_policy(V, (4,4))
    env.close()
