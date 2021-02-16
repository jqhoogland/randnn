import numpy as np

from ..topologies import get_gaussian_topology
from .base import BaseNN

class GaussianNN(BaseNN):
    def __init__(self,
                 coupling_strength: float = 1.,
                 **kwargs) -> None:
        """
        :param coupling_strength: See below.  Either coupling_matrix
            or coupling_strength must be provided, the former taking
            priority.
        :param kwargs: see parent class.
        """
        super().__init__(**kwargs)

        self.coupling_strength = coupling_strength

        # Binary matrix with (i, j) = 1 if there is an edge from i to j, 0 otherwise.
        self.edges_matrix = np.ones((self.n_dofs, self.n_dofs))

        # Weights matrix with a randomly drawn weight in every index.
        self.weights_matrix = get_gaussian_topology(
            self.n_dofs,
            coupling_strength,
            False,
            self.network_seed
        )

        # Final adjacency matrix where (i, j) = weight in weights matrix iff (i, j) == 1 in edges matrix
        # This is the element-wise product of the edges and weights matrices.
        self.coupling_matrix = np.multiply(self.edges_matrix, self.weights_matrix)


    def __repr__(self):
        return "<GaussianNN coupling_strength:{} n_dofs:{} timestep:{} seed: {}>".format(
            self.coupling_strength, self.n_dofs, self.timestep, self.network_seed)



# ------------------------------------------------------------
# TESTING


def test_jacobian_shape():
    coupling_matrix = np.eye(3, k=1)
    state = np.zeros(3)
    cont_nn = ContinuousNN(coupling_matrix=coupling_matrix, n_dofs=3)
    assert np.allclose(cont_nn.jacobian(state),
                       np.array([[-1, 1, 0], [0, -1, 1], [0, 0, -1]]))


def test_jacobian_saturation():
    coupling_matrix = np.eye(3, k=1)
    state = np.ones(3) * 100000000.
    cont_nn = ContinuousNN(coupling_matrix=coupling_matrix, n_dofs=3)
    assert np.allclose(cont_nn.jacobian(state), -np.eye(3))
