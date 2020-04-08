from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def create_plot(x, y, z, x_label, y_label, title):
    plt.scatter(x, y, c=z, cmap='cool')
    plt.colorbar()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def create_surface_plot(x, y, z, x_label, y_label, title):
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_trisurf(x, y, z, cmap='jet', linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def create_value_func_plot(V, size):
    plt.matshow(V.reshape(size),cmap='cool')
    plt.colorbar()
    plt.show()
