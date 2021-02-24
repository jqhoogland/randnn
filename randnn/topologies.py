"""

Contains the various network topologies that I'll be exploring.

Author: Jesse Hoogland
Year: 2020

"""
from typing import Optional

import numpy as np


def get_gaussian_topology(n_nodes: int, coupling_strength: float,
                          self_interaction: bool = False,
                          network_seed: Optional[int] =None) -> np.ndarray:
    """
    :param n_nodes: the number of nodes in the network
    :param coupling_strength: the final couplings are drawn from a
        normal distribution with variation $g^2/N$, where $g$ is the
        coupling strength and $N$ is the number of nodes.
    :param self_interaction: This is used if we randomly generatea
        coupling matrix.  It determines whether we do or do not allow
        diagonal elements on the connectivity matrix.
    :param network_seed: If we randomly generate a coupling matrix,
        this parameter determines the seed to use for np.random.  By
        default, this is left blank, so we do not specify a seed.
    """
    if network_seed:
        np.random.seed(network_seed)

    strength_normalized = (coupling_strength /
                           np.sqrt(n_nodes))

    normalized_matrix = np.random.normal(size=(n_nodes, n_nodes))

    if not self_interaction:
        diagonal = np.arange(n_nodes)
        normalized_matrix[diagonal, diagonal] = 0.

    # TODO: Go back to ignoring the diagonal in calculating the coupling strength
    coupling_matrix = (strength_normalized * normalized_matrix / np.std(normalized_matrix))

    return coupling_matrix


def test_get_gaussian_topology():
    coupling_matrix = get_gaussian_topology(10000, 10., self_interaction=True)
    assert np.isclose(np.std(coupling_matrix), 0.1, rtol=0.001)


def test_gaussian_topology_diagonal():
    coupling_matrix = get_gaussian_topology(10000, 10., self_interaction=False)
    assert np.allclose(np.diag(coupling_matrix), np.zeros(10000))
