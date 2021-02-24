import numpy as np

from randnn.networks.base import BaseNN
import numpy as np

from randnn.networks.base import BaseNN


def get_gaussian_weights(
        n_dofs: int, coupling_strength: float,
) -> np.ndarray:
    """
    :param n_dofs: the number of nodes in the network
    :param coupling_strength: the final couplings are drawn from a
        normal distribution with variation $g^2/N$, where $g$ is the
        coupling strength and $N$ is the number of nodes.
    """
    strength_normalized = (coupling_strength / np.sqrt(n_dofs))
    unit_matrix = np.random.normal(size=(n_dofs, n_dofs))

    coupling_matrix = (strength_normalized * unit_matrix / np.std(unit_matrix))

    return coupling_matrix


class GaussianNN(BaseNN):
    def __init__(self,
                 coupling_strength: float = 1.,
                 **kwargs) -> None:
        """
        :param coupling_strength: see ``get_gaussian_weights``
        :param kwargs: see parent class.
        """
        super().__init__(**kwargs)

        self.coupling_strength = coupling_strength

        # Binary matrix with (i, j) = 1 if there is an edge from i to j, 0 otherwise.
        self.edges_matrix = np.ones((self.n_dofs, self.n_dofs)) - np.eye(self.n_dofs)

        # Weights matrix with a randomly drawn weight in every index.
        self.weights_matrix = get_gaussian_weights(
            self.n_dofs,
            coupling_strength,
        )

        # Final adjacency matrix where (i, j) = weight in weights matrix iff (i, j) == 1 in edges matrix
        # This is the element-wise product of the edges and weights matrices.
        self.update_coupling_matrix(self.weights_matrix, self.edges_matrix)


    def __repr__(self):
        return "<GaussianNN coupling_strength:{} n_dofs:{} timestep:{} seed: {}>".format(
            self.coupling_strength, self.n_dofs, self.timestep, self.network_seed)



# ------------------------------------------------------------
# TESTING

def test_coupling_matrix():
    for g in np.arange(0.5, 20, 0.5):
        for n in range(50, 1000, 100):
            g_squiggle = g ** 2 / n
            coupling_matrix = GaussianNN(coupling_strength=g, n_dofs=n).coupling_matrix

            assert np.allclose(
                np.var(coupling_matrix),
                g_squiggle,
                rtol=1e-3
            ), "GaussianNN not initializing coupling matrix with correct weight dist."
