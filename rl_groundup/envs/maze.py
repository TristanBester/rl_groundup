# Created by Tristan Bester.
import os
import numpy as np

'''
Maze gridworld environment defined on page 135 of
"Reinforcement Learning: An Introduction."

Book reference:
Sutton, R. and Barto, A., 2014. Reinforcement Learning:
An Introduction. 1st ed. London: The MIT Press.
'''


CONST_RIGHT = 0
CONST_LEFT = 1
CONST_UP = 2
CONST_DOWN = 3

class Maze(object):
    def __init__(self):
        self.observation_space_size = 54
        self.action_space_size = 4
        self.wall_states = [7, 11, 16, 20, 25, 29, 41]
        self.start_state = 18
        self.terminal_state = 8


    def reset(self):
        self.agent_state = self.start_state
        return self.agent_state


    def __handle_action(self, action):
        if action == CONST_RIGHT:
            self.agent_state = self.agent_state+1 if self.agent_state % 9 != 8 \
                               and self.agent_state+1 not in self.wall_states \
                               else self.agent_state
        elif action == CONST_LEFT:
            self.agent_state = self.agent_state-1 if self.agent_state % 9 != 0 \
                               and self.agent_state-1 not in self.wall_states \
                               else self.agent_state
        elif action == CONST_UP:
            self.agent_state = self.agent_state-9 if self.agent_state > 8 and \
                               self.agent_state-9 not in self.wall_states else \
                               self.agent_state
        else:
            self.agent_state = self.agent_state+9 if self.agent_state < 45 and \
                               self.agent_state+9 not in self.wall_states else \
                               self.agent_state


    def get_predecessor_states(self, state):
        '''Return all states from which it is possible to transition into
        the specified state with one action.'''
        start_state = state
        self.agent_state = state
        preds = []
        for i in range(self.action_space_size):
            self.__handle_action(i)
            if start_state != self.agent_state:
                if i == CONST_RIGHT:
                    a = CONST_LEFT
                elif i == CONST_LEFT:
                    a = CONST_RIGHT
                elif i == CONST_UP:
                    a = CONST_DOWN
                else:
                    a = CONST_UP
                preds.append((self.agent_state, a))
            self.agent_state = start_state
        return preds


    def step(self, action):
        self.__handle_action(action)
        if self.agent_state == self.terminal_state:
            return self.agent_state, 1, True
        else:
            return self.agent_state, 0, False


    def print_char(self, ch, is_player):
        if is_player:
            print(f'\x1b[1;32;44m' + 'A' + '\x1b[0m', end='')
        elif ch == 'G':
            print(f'\x1b[1;32;44m' + ch + '\x1b[0m', end='')
        elif ch == '0':
            print(ch, end ='')
        else:
            print(f'\x1b[1;32;41m' + ch + '\x1b[0m', end='')


    def render(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        grid = ['0' if i not in self.wall_states else 'x' for i in\
                range(self.observation_space_size)]
        grid[self.terminal_state] = 'G'

        for i,x in enumerate(grid):
            if i % 9 == 8 and i != 0:
                self.print_char(x, i == self.agent_state)
                print()
            else:
                self.print_char(x, i == self.agent_state)
        print()
