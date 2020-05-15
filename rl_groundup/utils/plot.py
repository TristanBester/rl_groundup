# Created by Tristan Bester.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def create_line_plot(x, y, x_label, y_label, title):
    '''Create a line plot using the given data.'''
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def create_bar_plot(x, y, x_label, y_label, title):
    '''Create a bar plot using the given data.'''
    plt.bar(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def create_surface_plot(x, y, z, x_label, y_label, title):
    '''Create a surface plot using the given data.'''
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_trisurf(x, y, z, cmap='jet', linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def create_value_func_plot(V, size, title):
    '''Create a plot to show the value of each state in a
    grid world problem.'''
    plt.matshow(V.reshape(size),cmap='cool')
    fig = plt.gcf()
    fig.set_size_inches((7,5))
    plt.title(title)
    plt.colorbar()
    plt.show()


def plot_blackjack_value_functions(V):
    '''Plot a value function for when the player does and does not
    have a usable ace in their hand.'''
    keys = np.array(list(V.keys()))
    keys_u = keys[keys[:,2] == 1]
    keys_n = keys[keys[:,2] != 1]
    z_u = [V[tuple(s)] for s in keys_u]
    z_n = [V[tuple(s)] for s in keys_n]
    create_surface_plot(keys_u[:, 0], keys_u[:, 1], z_u, \
                        'Hand:','Dealer:','Usable ace:')
    create_surface_plot(keys_n[:, 0], keys_n[:, 1], z_n, \
                        'Hand:','Dealer:','No usable ace:')


def plot_mountain_car_value_function(min_x, max_x, min_y, max_y, v, tile_coder):
    '''Create a surface plot illustrating the value of each state
    under a specific policy.'''
    x = np.linspace(min_x, max_x, 20)
    y = np.linspace(min_y, max_y, 20)

    X,Y = np.meshgrid(x,y)
    all_states = np.array([[x,y] for x,y in zip(X.ravel(),Y.ravel())])
    all_feature_vectors = np.array([tile_coder.get_tile_code(s) for s in all_states])
    Z = np.array([-1 * v.evaluate(x) for x in all_feature_vectors])
    Z = Z.reshape(X.shape)
    create_surface_plot(X.ravel(), Y.ravel(), Z.ravel(), 'Position:', \
    'Velocity:', 'Mountain car value function::')
