"""

Contains basic functions for plotting various elements of this project:
- (low-dimensional projections of) trajectories and their averages
- Lyapunov spectra
- etc.

Author: Jesse Hoogland
Year: 2020

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, rc

plt.rc('text', usetex=True)


def plot_trajectory_avg(trajectory):
    avg_trajectory = np.mean(trajectory, axis=1)
    stdev_trajectory = np.std(trajectory, axis=1)
    plt.errorbar(np.arange(avg_trajectory.size),
                 avg_trajectory,
                 yerr=stdev_trajectory)
    plt.show()


def plot_random_matrix_spectrum(matrix, radius=None):
    N, _ = matrix.shape  # Assuming square
    if radius is None:
        radius = np.std(matrix) * np.sqrt(N)

    C = np.zeros((N, N), dtype=np.complex_)
    eigs, _ = np.linalg.eig(matrix + C)

    #print(eigs)
    fix, ax = plt.subplots()
    ax.scatter(eigs.real, eigs.imag, c=(0, 0, 0), s=2, alpha=0.5)
    circle = patches.Circle((0, 0),
                            radius,
                            fill=False,
                            color="k",
                            linewidth=2,
                            edgecolor='b')

    plt.axis("square")
    ax.add_patch(circle)
    ax.set_title("Eigenvalue spectrum of the connectivity matrix $J_{ij}$")
    ax.set_ylabel("Complex part")
    ax.set_xlabel("Real part")
    plt.show()
