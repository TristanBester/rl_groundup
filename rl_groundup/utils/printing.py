# Created by Tristan Bester.
import sys
import numpy as np


def print_episode(episode, n_episodes):
    '''Allows for readable output during training.'''
    # Back to previous line.
    sys.stdout.write("\033[F")
    # Clear line.
    sys.stdout.write("\033[K")
    n_bars = int((episode/n_episodes) * 25)
    print(f'Current episode: {episode}\t[{"=" * n_bars}{"#" * (25-n_bars)}]')


def print_grid_world_actions(policy, shape, terminal_states):
    '''Print the actions taken by the given policy in a gridworld problem.'''
    out = np.full(len(policy), 'T')
    for i,a in enumerate(policy):
        if a == 0:
            a = '>'
        elif a == 1:
            a = '<'
        elif a == 2:
            a = '^'
        else:
            a = 'V'
        out[i] = a
    out[terminal_states] = 'T'
    print(f'Actions in deterministic optimal policy: \n{out.reshape(shape[0], shape[1])}')
