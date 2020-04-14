import sys
import numpy as np

def print_episode(episode, n_episodes):
    '''Readable output during trainining.'''
    sys.stdout.write("\033[F") # Back to previous line.
    sys.stdout.write("\033[K") # Clear line.
    n_bars = int((episode/n_episodes) * 25)
    print(f'Current episode: {episode}\t[{"=" * n_bars}{"#" * (25-n_bars)}]')

def print_grid_world_actions(policy):
    '''Print the actions taken by the given policy in the grid world problem.'''
    out = np.full(16, 'T')
    for i,x in enumerate(policy[1:-1]):
        a = np.argmax(x)
        if a == 0:
            a = '>'
        elif a == 1:
            a = '<'
        elif a == 2:
            a = '^'
        else:
            a = 'V'
        out[i+1] = a
    print(f'Actions in deterministic optimal policy: \n{out.reshape(4,4)}')
