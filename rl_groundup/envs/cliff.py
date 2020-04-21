import os
import numpy as np

RIGHT = 0
LEFT = 1
UP = 2
DOWN = 3

class CliffWalking(object):
    def __init__(self):
        self.observation_space_size = 48
        self.action_space_size = 4
        self.terminal_state = 47
        self.cliff_states = list(range(37,47))


    def reset(self):
        self.agent_state = 36
        return self.agent_state


    def handle_action(self, action):
        if action == RIGHT:
            self.agent_state = self.agent_state+1 if self.agent_state%12 != 11 \
                               else self.agent_state
        elif action == LEFT:
            self.agent_state = self.agent_state-1 if self.agent_state%12 != 0 \
                               else self.agent_state
        elif action == UP:
            self.agent_state = self.agent_state-12 if self.agent_state > 11 \
                               else self.agent_state
        else:
            self.agent_state = self.agent_state+12 if self.agent_state < 36 \
                               else self.agent_state


    def step(self, action):
        start_state = self.agent_state
        self.handle_action(action)
        reward = -1
        done = False
        if self.agent_state in self.cliff_states:
            reward = -100
            self.reset()
        elif self.agent_state == start_state:
            reward = -100
        elif self.agent_state == self.terminal_state:
            reward = 100
            done = True
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
        grid = ['x'] * 37 +  ['o'] * 10 + ['x']

        for i,x in enumerate(grid):
            if i % 12 == 11 and i != 0:
                self.print_char(x, i == self.agent_state)
                print()
            else:
                self.print_char(x, i == self.agent_state)
        print()
