import numpy as np

from randnn.networks.base import BaseNN


def get_exponential_weights(n_dofs: int, coupling_strength: float) -> np.ndarray:
    """
    $J_{ij}$ is drawn from $f(x; 1/\beta) = \exp(-x / \beta) / \beta$

    :param n_dofs: the number of nodes in the network
    :param coupling_strength: the scale parameter (equiv. the inverse rate).
        the final couplings are drawn from an exponential distribution with variation $g^2/N$, where $g$ is the
        coupling strength and $N$ is the number of nodes.
    """
    strength_normalized = (coupling_strength / np.sqrt(n_dofs))
    coupling_matrix = np.random.exponential(size=(n_dofs, n_dofs), scale=coupling_strength)
    coupling_matrix *= np.random.choice([1, -1], size=(n_dofs, n_dofs))  # random sign for each node

    coupling_matrix *= (strength_normalized / np.std(coupling_strength))

    return coupling_matrix


class ExponentialNN(BaseNN):
    def __init__(self,
                 coupling_strength: float = 1.,
                 **kwargs) -> None:
        """
        :param coupling_strength: See ``get_exponential_weights``
        :param kwargs: see parent class.
        """
        super().__init__(**kwargs)

        self.coupling_strength = coupling_strength

        # Binary matrix with (i, j) = 1 if there is an edge from i to j, 0 otherwise.
        self.edges_matrix = np.ones((self.n_dofs, self.n_dofs)) - np.eye(self.n_dofs)

        # Weights matrix with a randomly drawn weight in every index.
        self.weights_matrix = get_exponential_weights(
            self.n_dofs,
            coupling_strength
        )

        # Final adjacency matrix where (i, j) = weight in weights matrix iff (i, j) == 1 in edges matrix
        # This is the element-wise product of the edges and weights matrices.
        self.update_coupling_matrix(self.weights_matrix, self.edges_matrix)

    def __repr__(self):
        return "<ExponentialNN coupling_strength:{} n_dofs:{} timestep:{} seed: {}>".format(
            self.coupling_strength, self.n_dofs, self.timestep, self.network_seed)

# ------------------------------------------------------------
# TESTING
