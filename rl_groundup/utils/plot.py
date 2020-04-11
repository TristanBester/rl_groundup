from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def create_line_plot(x, y, x_label, y_label, title):
    '''Create a line plot using given data.'''
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

def create_surface_plot(x, y, z, x_label, y_label, title):
    '''Create a surface plot using given data.'''
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_trisurf(x, y, z, cmap='jet', linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def create_value_func_plot(V, size, title):
    '''Create a plot to show the value of each state in the
    grid world problem.'''
    plt.matshow(V.reshape(size),cmap='cool')
    fig = plt.gcf()
    fig.set_size_inches((7,5))
    plt.title(title)
    plt.colorbar()
    plt.show()
