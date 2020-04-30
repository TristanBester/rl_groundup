# Created by Tristan Bester.
import re
import os
import time
import numpy as np
from itertools import product

'''
Racetrack environment defined on page 91 of
"Reinforcement Learning: An Introduction."

Book reference:
Sutton, R. and Barto, A., 2014. Reinforcement Learning:
An Introduction. 1st ed. London: The MIT Press.
'''


class RaceTrack(object):
    def __init__(self, use_noise=False):
        self.__init_grid()
        self.delta_max = 3
        self.delta_min = 0
        self.use_noise = use_noise
        self.action_space = list(product(range(-1, 2), range(-1, 2)))
        self.state_space = [(2, 17),(2, 18),(1, 19),(2, 20), (1, 10),
                            (7, 9),(8, 8),(7, 7),(3, 6)]
        self.observation_space_size = 333
        self.action_space_size = 9


    def __init_grid(self):
        track = np.full((33, 20), 0)
        specifications = [(2, 17, 0), (2, 18, 0),( 1, 19, 0), (2, 20, 0),
                          (1, 10, 1), (7, 9, 1), (8, 8, 1), (7, 7, 1), (3, 6, 1)]
        curr_row = 0
        for spec in specifications:
            n, w, left = spec
            for row in range(curr_row, curr_row + n):
                if left:
                    track[row,10-w: 10] = 1
                else:
                    track[row,20 - w:] = 1
            curr_row += n

        # Set end states.
        track[:7,-1] = -1
        self.track = track


    def reset(self):
        self.state = np.array([32 ,np.random.randint(4,10)], dtype='int')
        # Velocity format: [dy, dx].
        self.velocity = np.zeros(2,dtype='int')
        state = ((self.state[0], self.state[1]), (self.velocity[0], self.velocity[1]))
        return self.adjust_col(state)


    def __validate_dy(self):
        # Agent must always move in negative direction: dy constrained to
        # be within interval [-3, 0].
        if self.velocity[0] > self.delta_min:
            self.velocity[0] = self.delta_min
        elif self.velocity[0] < -self.delta_max:
            self.velocity[0] = -self.delta_max


    def __validate_dx(self):
        # Agent horizontal velocity constrained to be within interval [-2,2].
        if self.velocity[1] < -2 or self.velocity[1] > 2:
            self.velocity[1] = np.sign(self.velocity[1]) * 2


    def __update_velocity(self, dy, dx):
        if self.use_noise and np.random.uniform() > 0.1:
            self.velocity += [dy,dx]
        else:
            self.velocity += [dy,dx]
        self.__validate_dy()
        self.__validate_dx()


    def adjust_col(self, state):
        '''Adjust state to remove the edges of the grid.'''
        pos = state[0]
        y = pos[0]
        vel = state[1]

        if y  < 2:
            return ((y, pos[1] - 3), vel)
        elif y < 4:
            return ((y, pos[1] - 2), vel)
        elif y < 5:
            return ((y, pos[1] - 1), vel)
        elif y > 7 and y < 15:
            return ((y, pos[1] - 1), vel)
        elif y > 14 and y < 23:
            return ((y, pos[1] - 2), vel)
        elif y > 22 and y < 30:
            return ((y, pos[1] - 3), vel)
        elif y > 29 and y < 33:
            return ((y, pos[1] - 4), vel)
        else:
            return state


    def __get_single_steps(self, dy, dx):
        '''Decompose full movement into single state transitions.'''
        vert = np.ones(abs(dy))
        hor = np.ones(abs(dx))
        diff = abs(len(vert) - len(hor))

        if len(hor) < len(vert):
            hor = np.append(hor, np.zeros(diff))
        else:
            vert = np.append(vert, np.zeros(diff))

        hor = np.sign(dx) * hor
        vert = np.sign(dy) * vert
        return np.c_[vert.reshape(-1,1), hor.reshape(-1,1)].astype('int')


    def __step_valid(self, dy, dx):
        actions = self.__get_single_steps(dy, dx)
        start_state = self.state

        x_invalid = lambda x: True if x < 0 or x > 19 else False
        y_invalid = lambda y: True if y < 0 or y > 32 else False

        # Move agent in single steps along path. Test if trajectory
        # intersects wall after each transition.
        for a in actions:
            start_state = start_state + a
            self.state += a
            if (x_invalid(start_state[1]) or
                y_invalid(start_state[0]) or
                self.track[start_state[0], start_state[1]] == 0):
                return False
        return True


    def step(self, action):
        done = False
        reward = -1
        dy, dx = self.action_space[action]
        self.__update_velocity(dy,dx)

        if not self.__step_valid(self.velocity[0], self.velocity[1]):
            state = self.reset()
            return state, -1, False
        else:
            state = ((self.state[0], self.state[1]), (self.velocity[0], self.velocity[1]))
            state = self.adjust_col(state)

            # If state terminal done set to True.
            if self.track[self.state[0], self.state[1]] == -1:
                reward = 0
                done = True

            return state, reward, done


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
