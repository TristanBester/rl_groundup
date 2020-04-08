import numpy as np

RIGHT = 0
LEFT = 1
UP = 2
DOWN = 3

class GridWorld(object):
    def __init__(self):
        self.states = np.zeros(16)
        self.terminal_states = [0, 15]
        self.observation_space_size = 16
        self.action_space_size = 4

    def reset(self):
        self.agent_state = 5
        return self.agent_state

    def get_new_state(self, action):
        # Terinal states cannot be exited.
        if self.agent_state in self.terminal_states:
            return (curr_state, 0, True)

        if action == RIGHT:
            if self.agent_state % 4 == 3:
                self.agent_state = self.agent_state
            else:
                self.agent_state = self.agent_state + 1
        elif action == LEFT:
            if self.agent_state % 4 == 0:
                self.agent_state = self.agent_state
            else:
                self.agent_state = self.agent_state - 1
        elif action == UP:
            if self.agent_state < 4:
                self.agent_state = self.agent_state
            else:
                self.agent_state = self.agent_state - 4
        else:
            if self.agent_state > 11:
                self.agent_state = self.agent_state
            else:
                self.agent_state = self.agent_state + 4

        return self.agent_state

    def step(self, action):
        n_state = self.get_new_state(action)
        if n_state in self.terminal_states:
            return (n_state, 0, True)
        else:
            return (n_state, -1, False)
