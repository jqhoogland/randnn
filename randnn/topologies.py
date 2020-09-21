"""

Contains the various network topologies that I'll be exploring.

Author: Jesse Hoogland
Year: 2020

"""
import numpy as np
import matplotlib.pyplot as plt


def get_gaussian_topology(n_nodes: int,
                          coupling_strength: float,
                          self_interaction: bool = False):
    """
    :param n_nodes: the number of nodes in the network
    :param coupling_strength: the final couplings are drawn from a normal distribution
    with variation $g^2/N$, where $g$ is the coupling strength and $N$ is the number of nodes.
    :param self_interaction:

    """
    coupling_matrix = np.random.normal(scale=coupling_strength /
                                       np.sqrt(n_nodes),
                                       size=(n_nodes, n_nodes))

    if not self_interaction:
        diagonal = np.arange(n_nodes)
        coupling_matrix[diagonal, diagonal] = 0.

    return coupling_matrix


def plot_random_matrix_spectrum(matrix):
    N, _ = matrix.shape  # Assuming square
    radius = np.std(matrix) * np.sqrt(N)
    eigs, _ = np.linalg.eigh()

    fix, ax = plt.subplots
    ax.plot(eigs.real, eigs.imag)
    circle = plt.Circle((0, 0), radius, color=None, edgecolor='b')
    ax.add_artist(circle)
    plt.show()
