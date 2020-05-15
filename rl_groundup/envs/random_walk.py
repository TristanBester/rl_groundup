# Created by Tristan Bester.
import numpy as np

'''
Random Walk Markov Reward Process (MRP) defined on page 102
of "Reinforcement Learning: An Introduction."

Book reference:
Sutton, R. and Barto, A., 2014. Reinforcement Learning:
An Introduction. 1st ed. London: The MIT Press.
'''


class RandomWalk(object):
    def __init__(self, n_states):
        # Ensure odd number of states.
        self.n_states = n_states if n_states % 2 == 1 else n_states + 1
        self.start_state = self.n_states//2


    def reset(self):
        self.agent_state = self.start_state
        return self.agent_state


    def step(self):
        self.agent_state += np.random.choice([-1,1])
        reward = 0
        done = False
        if self.agent_state == 0:
            done = True
        elif self.agent_state == self.n_states-1:
            done = True
            reward = 1
        return self.agent_state, reward, done
