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


def test_get_gaussian_topology():
    coupling_matrix = get_gaussian_topology(10000, 10., self_interaction=True)
    assert np.isclose(np.std(coupling_matrix), 0.1, rtol=0.001)


def test_gaussian_topology_diagonal():
    coupling_matrix = get_gaussian_topology(10000, 10., self_interaction=False)
    assert np.allclose(np.diag(coupling_matrix), np.zeros(10000))
