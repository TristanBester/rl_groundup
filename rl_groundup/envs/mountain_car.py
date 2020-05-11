# Created by Tristan Bester.
import numpy as np

'''
Mountain car environment defined on page 198
of "Reinforcement Learning: An Introduction."

Book reference:
Sutton, R. and Barto, A., 2014. Reinforcement Learning:
An Introduction. 1st ed. London: The MIT Press.
'''

class MountainCar(object):
    def __init__(self, max_steps=None):
        self.min_pos = -1.2
        self.max_pos = 0.5
        self.max_speed = 0.07
        self.goal_pos = 0.4
        self.init_pos_range = [-0.6, -0.5]
        self.action_space_size = 3
        self.max_steps = max_steps


    def reset(self):
        self.steps = 1
        self.velocity = 0
        self.position = np.random.uniform(self.init_pos_range[0],
                                          self.init_pos_range[1])
        return (self.position, self.velocity)


    def handle_velocity(self, action):
        self.velocity = self.velocity + 0.001 * action - \
                        0.0025 * np.cos(3 * self.position)
        if abs(self.velocity) > self.max_speed:
            self.velocity = np.sign(self.velocity) * self.max_speed


    def handle_position(self):
        self.position = self.position + self.velocity
        if self.position < -1.2:
            self.position = 0


    def step(self, action):
        self.steps += 1
        self.handle_velocity(action)
        self.handle_position()
        done = self.position >= self.goal_pos
        reward = -1 if not done else 0
        if self.max_steps is not None and self.steps > self.max_steps:
            return (self.position, self.velocity), reward, True
        else:
            return (self.position, self.velocity), reward, done
