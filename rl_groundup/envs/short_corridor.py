import numpy as np

LEFT = 0

class ShortCorridor(object):
    def __init__(self):
        self.observation_space_size = 4
        self.action_space_size = 2

    def reset(self):
        self.agent_state = 0
        return self.agent_state

    def handle_action(self, action):
        if action == LEFT:
            if self.agent_state == 0:
                self.agent_state = 0
            elif self.agent_state == 1:
                self.agent_state += 1
            else:
                self.agent_state -= 1
        else:
            if self.agent_state == 0:
                self.agent_state += 1
            elif self.agent_state == 1:
                self.agent_state -= 1
            else:
                self.agent_state += 1

    def step(self, action):
        self.handle_action(action)
        done = self.agent_state == 3
        reward = -1 if not done else 0
        return self.agent_state, reward, done
