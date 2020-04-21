import os
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


    def reset(self):
        self.agent_state = np.random.choice([3,6,9,12])
        return self.agent_state


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


    def handle_action(self, a):
        if a == RIGHT:
            self.agent_state = self.agent_state + 1 if self.agent_state % 4 \
                               != 3 else self.agent_state
        elif a == LEFT:
            self.agent_state = self.agent_state - 1 if self.agent_state % 4 \
                               != 0 else self.agent_state
        elif a == UP:
            self.agent_state = self.agent_state - 4 if self.agent_state > 3 \
                               else self.agent_state
        else:
            self.agent_state = self.agent_state + 4 if self.agent_state < 12 \
                               else self.agent_state



    def step(self, action):
        start_state = self.agent_state
        self.handle_action(action)
        if self.agent_state == start_state:
            reward = -10
        elif self.agent_state in self.terminal_states:
            reward = 0
        else:
            reward = -1
        done = reward == 0
        return self.agent_state, reward, done


    def print_char(self, ch, is_player):
        if is_player:
            print(f'\x1b[1;32;44m' + ch + '\x1b[0m', end='')
        elif ch == 'x':
            print(ch, end ='')
        else:
            print(f'\x1b[1;32;41m' + ch + '\x1b[0m', end='')


    def render(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        grid = ['T'] + ['x'] * 14 + ['T']

        for i,x in enumerate(grid):
            if i % 4 == 3 and i != 0:
                self.print_char(x, i == self.agent_state)
                print()
            else:
                self.print_char(x, i == self.agent_state)
        print()
