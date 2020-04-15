import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import time
import re
import os

class RaceTrack(object):
    def __init__(self, use_noise=False):
        self.__init_grid()
        self.delta_max = 3
        self.delta_min = 0
        self.use_noise = use_noise
        self.action_space = list(product(range(-1, 2), range(-1, 2)))
        self.state_space = [(2, 17),(2, 18),(1, 19),(2, 20), (1, 10),
                            (7, 9),(8, 8),(7, 7),(3, 6)]


    def __init_grid(self):
        track = np.full((33, 20), 0)
        specifications = [(2, 17, 0),(2, 18, 0),(1, 19, 0),(2, 20, 0),
                          (1, 10, 1),(7, 9, 1),(8, 8, 1),(7, 7, 1),
                          (3, 6, 1)]
        curr_row = 0
        for spec in specifications:
            n, w, left = spec
            for row in range(curr_row, curr_row + n):
                if left:
                    track[row,10-w: 10] = 1
                else:
                    track[row,20 - w:] = 1
            curr_row += n
        # Set start and end states.
        #track[-1,4:10] = -1
        track[:7,-1] = -1
        self.track = track


    def reset(self):
        self.state = np.array([32 ,np.random.randint(4,10)], dtype='int')
        #dy, dx
        self.velocity = np.zeros(2,dtype='int')
        return ((self.state[0], self.state[1]), (self.velocity[0], self.velocity[1]))


    def __validate_dy(self):
        # Agent must always move in negative direction, dy contrained [0, -3].
        if self.velocity[0] > self.delta_min:
            self.velocity[0] = self.delta_min
        elif self.velocity[0] < -self.delta_max:
            self.velocity[0] = -self.delta_max


    def __validate_dx(self):
        # Agent horizontal vel contrained to [-5,5]
        if self.velocity[1] < -2 or self.velocity[1] > 2:
            self.velocity[1] = np.sign(self.velocity[1]) * 2


    def __update_velocity(self, dy, dx):
        if self.use_noise and np.random.uniform() > 0.1:
            self.velocity += [dy,dx]
        else:
            self.velocity += [dy,dx]
        self.__validate_dy()
        self.__validate_dx()


    def step(self, action):
        done = False
        reward = -1
        dy, dx = self.action_space[action]
        self.__update_velocity(dy,dx)
        self.state += self.velocity

        x_invalid = lambda x: True if x < 0 or x > 19 else False
        y_invalid = lambda y: True if y < 0 or y > 32 else False

        if (x_invalid(self.state[1])
            or y_invalid(self.state[0])
            or not self.track[self.state[0], self.state[1]]):
            self.reset()
        elif self.track[self.state[0], self.state[1]] == -1:
            reward = 0
            done = True
        return ((self.state[0], self.state[1]), (self.velocity[0], \
                 self.velocity[1])), reward, done


    def __print_col(self, line, idx, col=['1','32','41'], is_start=False):
        line = list(line)
        for x in range(len(line)):
            if x in idx:
                print(f'\x1b[{col[0]};{col[1]};{col[2]}m' + line[x] + '\x1b[0m', end='')
            elif x == len(line) -1 and line[x] == '1':
                print(f'\x1b[1;32;44m' + line[x] + '\x1b[0m', end='')
            elif is_start and line[x] == '1':
                print(f'\x1b[1;32;44m' + line[x] + '\x1b[0m', end='')
            else:
                print(line[x], end='')
        print()


    def render(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        str = np.array2string(self.track)
        str = re.sub('[^0-9,\n]', '', str)
        lines = str.split('\n')

        for i in range(len(lines)):
            if i == self.state[0]:
                self.__print_col(lines[i], [self.state[1]], is_start=(i==32))
            elif i == 32:
                self.__print_col(lines[i], [4,5,6,7,8,9], col=['1','32','44'], is_start=False)
            elif i < 7:
                self.__print_col(lines[i], [19], col=['1','32','44'])
            else:
                print(lines[i])
