# Created by Tristan Bester.
import os
import numpy as np

'''
Windy gridworld environment defined on page 106 of
"Reinforcement Learning: An Introduction."

Book reference:
Sutton, R. and Barto, A., 2014. Reinforcement Learning:
An Introduction. 1st ed. London: The MIT Press.
'''


CONST_RIGHT = 0
CONST_LEFT = 1
CONST_UP = 2
CONST_DOWN = 3

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
        if action == CONST_RIGHT:
            return 1 if self.agent_state % 10 != 9 else 0
        elif action == CONST_LEFT:
            return -1 if self.agent_state % 10 != 0 else 0
        elif action == CONST_UP:
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
            reward = -100000
        else:
            reward = -1
        done = reward == 10
        return self.agent_state, reward, done


    def print_char(self, ch, is_player):
        if is_player:
            print(f'\x1b[1;32;44m' + 'A' + '\x1b[0m', end='')
        elif ch == 'G':
            print(f'\x1b[1;31;46m' + ch + '\x1b[0m', end='')
        else:
            print(ch, end ='')


    def render(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        grid = ['0' for i in range(self.observation_space_size)]
        grid[self.terminal_state] = 'G'

        for i,x in enumerate(grid):
            if i % 10 == 9 and i != 0:
                self.print_char(x, i == self.agent_state)
                print()
            else:
                self.print_char(x, i == self.agent_state)
        print()
