"""

Contains the various network topologies that I'll be exploring.

Author: Jesse Hoogland
Year: 2020

"""
import numpy as np

class NetworkTopology(object):
    def __init__(self, n_nodes, coupling_matrix, init_state=None):
        self.n_nodes = n_nodes
        self.coupling_matrix = coupling_matrix

        if (init_state is None):
            init_state = np.random.uniform(n_nodes)

        self.state = init_state

    def update(self, state):
        self.state = state

class GaussianRandomTopology(NetworkTopology):
    def __init__(self, n_nodes, coupling, self_interaction=False, init_state=None):
        """
        :param n_nodes: the number of nodes in the network
        :param coupling: the final couplings are drawn from a normal distribution
        with variation $g^2/N$, where $g$ is the coupling and $N$ is the number of nodes.
        :param self_interaction:

        """
        coupling_matrix = np.random.normal(scale=coupling ** 2 / n_nodes, size=(n_nodes, n_nodes))

        if not self_interaction:
            diagonal = np.arange(n_nodes)
            coupling_matrix[diagonal, diagonal] = 0.

        super(GaussianRandomTopology, self).__init__(n_nodes, coupling_matrix, init_state)
