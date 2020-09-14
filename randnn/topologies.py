"""

Contains the various network topologies that I'll be exploring.

Author: Jesse Hoogland
Year: 2020

"""
import numpy as np


def get_gaussian_topology(n_nodes: int,
                          coupling: float,
                          self_interaction: bool = False):
    """
    :param n_nodes: the number of nodes in the network
    :param coupling: the final couplings are drawn from a normal distribution
    with variation $g^2/N$, where $g$ is the coupling and $N$ is the number of nodes.
    :param self_interaction:

    """
    coupling_matrix = np.random.normal(scale=coupling**2 / n_nodes,
                                       size=(n_nodes, n_nodes))

    if not self_interaction:
        diagonal = np.arange(n_nodes)
        coupling_matrix[diagonal, diagonal] = 0.

    return coupling_matrix
