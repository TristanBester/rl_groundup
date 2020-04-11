import numpy as np

RIGHT = 0
LEFT = 1
UP = 2
DOWN = 3

class GridWorld(object):
    def __init__(self):
        self.observation_space_size = 16
        self.action_space_size = 4
        self.terminal_states = [0,15]
        self.init_transition_matrix()

    def init_transition_matrix(self):
        self.P = {}
        for state in range(self.observation_space_size):
            for a in range(self.action_space_size):
                if state in self.terminal_states:
                    self.P[state,a] = (1,state,0,True)
                else:
                    if a == RIGHT:
                        n_state = state + 1 if state % 4 != 3 else state
                    elif a == LEFT:
                        n_state = state - 1 if state % 4 != 0 else state
                    elif a == UP:
                        n_state = state - 4 if state > 3 else state
                    else:
                        n_state = state + 4 if state < 12 else state
                    self.P[state,a] = (1, n_state, -1, n_state in self.terminal_states)
