import numpy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def plot_dipoles(dipoles):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    n = 100
    ax.scatter(dipoles[:, 0], dipoles[:, 1], dipoles[:, 2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([numpy.min(dipoles[:, 0]), numpy.max(dipoles[:, 0])])
    ax.set_ylim([numpy.min(dipoles[:, 1]), numpy.max(dipoles[:, 1])])
    ax.set_zlim([numpy.min(dipoles[:, 2]), numpy.max(dipoles[:, 2])])

    plt.show(block=True)