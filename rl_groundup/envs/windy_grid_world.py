import numpy as np

RIGHT = 0
LEFT = 1
UP = 2
DOWN = 3

class WindyGridWorld(object):
    def __init__(self):
        self.action_space_size = 4
        self.observation_space_size = 70
        self.col_wind_vals = [0,0,0,-10,-10,-10,-20,-20,-10,0]
        self.start_state = 30
        self.terminal_state = 37


    def reset(self):
        self.agent_state = self.start_state
        return self.agent_state


    def step_delta(self, action):
        if action == RIGHT:
            return 1 if self.agent_state % 10 != 9 else 0
        elif action == LEFT:
            return -1 if self.agent_state % 10 != 0 else 0
        elif action == UP:
            return -10 if self.agent_state > 9  else 0
        else:
            return 10 if self.agent_state < 60 else 0


    def wind_delta(self):
        col = self.agent_state % 10
        return self.col_wind_vals[col]


    def handle_action(self, action):
        step = self.step_delta(action)
        wind = self.wind_delta()
        pos = self.agent_state + step + wind
        self.agent_state = pos if pos > -1 else (pos%10)


    def step(self, action):
        start_state = self.agent_state
        self.handle_action(action)
        if self.agent_state == self.terminal_state:
            reward = 10
        elif start_state == self.agent_state:
            #reward = -2
            reward = -100000
        else:
            reward = -1
        done = reward == 10
        return self.agent_state, reward, done
