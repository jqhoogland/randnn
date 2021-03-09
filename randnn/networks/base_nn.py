"""

Contains the various networks I'll be exploring,
i.e. the pairs of neuronal dynamics and network topologies.

Author: Jesse Hoogland
Year: 2020

"""
from typing import Optional

import numpy as np

from pynamics.trajectories import DeterministicTrajectory
from ..plotting import plot_coupling_matrix_spectrum
from ..topologies import get_fully_connected_edges


class BaseNN(DeterministicTrajectory):
    def __init__(self,
                 network_seed: Optional[int] = None,
                 n_dofs: int = 100,
                 **kwargs) -> None:
        """
        :param network_seed: If we randomly generate a coupling
            matrix, this parameter determines the seed to use for
            np.random.  This is useful if we'd like to compare similar
            networks for different values of the coupling_strength.
            By default, this is left blank, so we do not specify a
            seed.
        :param kwargs: see parent class.
        """
        self.network_seed = network_seed
        if network_seed:
            np.random.seed(network_seed)

        # To make accessing non-diagonal elements easier
        self.non_diagonal_idxs = np.where(np.ones((n_dofs, n_dofs)) - np.eye(n_dofs))

        super().__init__(n_dofs=n_dofs, **kwargs)

    def __repr__(self):
        return "<BaseNN n_dofs:{} timestep:{} seed: {}>".format(
            self.n_dofs, self.timestep, self.network_seed)

    @property
    def coupling_radius(self):
        return None

    @staticmethod
    def activation(state: np.ndarray) -> np.ndarray:
        return np.tanh(state)

    @staticmethod
    def activation_prime(state: np.ndarray) -> np.ndarray:
        return np.power(1. / np.cosh(state), 2)

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        return -(1 - self.timestep) * np.eye(
            self.n_dofs) + self.timestep * np.diag(
            self.activation_prime(state)) @ self.coupling_matrix

    def take_step(self, t: int, state: np.ndarray) -> np.ndarray:
        return -state + self.coupling_matrix @ self.activation(state)

    def plot_spectrum(self):
        assert self.coupling_matrix is not None
        plot_coupling_matrix_spectrum(self.coupling_matrix, self.coupling_radius)


class ElementWiseInit:
    """
    A mixin for initializing the coupling matrix according to:
    :param coupling_matrix ($J$): Given by $J = S \odot E \odot W + M$ (i.e., elementwise multiplication)

    :param signs_matrix ($S$): Element (i, j) is the sign multiplier of edge (i, j) in the final matrix.
    :param edges_matrix ($E$): Element (i, j) is a binary whether to include edge (i, j) in the final matrix.
    :param weights_matrix ($W$): Element (i, j) has the magnitude of the weight of edge (i, j), ignoring offset.
    :param offset_matrix ($M$): Element (i, j) is an offset for the weight of edge (i, j) in the final matrix.
    """

    def __init__(self, **kwargs):
        # Signs generates a sign multiplier for every possible edge.
        # By default this is a matrix of ones (i.e. do not change the edge weights).
        # This becomes important for Dale's law & one-sided edge-weight distributions.
        self.signs_matrix = self.gen_signs()

        # Binary matrix with (i, j) = 1 if there is an edge from i to j, 0 otherwise.
        self.edges_matrix = self.gen_edges()

        # Weights matrix generates a weight for *every* possible pair of nodes
        self.weights_matrix = self.gen_weights()

        # Offset is a bias on each edge
        self.offset_matrix = self.gen_offset()

        # Final adjacency matrix where (i, j) = weight in weights matrix iff (i, j) == 1 in edges matrix
        # This is the element-wise product of the edges and weights matrices.
        self.coupling_matrix = self.compute_coupling_matrix(self.weights_matrix, self.edges_matrix, self.signs_matrix,
                                                            self.offset_matrix)

        super().__init__(**kwargs)

    def gen_edges(self):
        return get_fully_connected_edges(self.n_dofs, False)

    def gen_weights(self):
        raise NotImplementedError

    def gen_signs(self):
        return np.ones((self.n_dofs, self.n_dofs))

    def gen_offset(self):
        return 0

    @staticmethod
    def _compute_coupling_matrix(weights_matrix, edges_matrix=1, signs_matrix=1, offset_matrix=0):
        return np.multiply(np.multiply(weights_matrix, edges_matrix), signs_matrix) + offset_matrix

    def compute_coupling_matrix(self, weights_matrix, edges_matrix=1, signs_matrix=1, offset_matrix=0):
        return self._compute_coupling_matrix(weights_matrix, edges_matrix, signs_matrix, offset_matrix)


class MatrixInit:
    """
    Based on Ahmadian et al. [@ahmadian2015]

    :param coupling_matrix ($J$): The final coupling matrix.
            Given by $$J= \Sigma A P + M$$
    :param randomness_matrix ($A$): a normally distributed matrix of zero mean and unit variance,
    :param variances_matrix ($\Sigma$): the variances of the final edges.
    :param projection_matrix ($P$): enforces any extra constraints.
    :param offset_matrix ($M$): the offset of the final matrix.
    """

    def __init__(self, **kwargs):
        self.variances_matrix = self.gen_variances()
        self.randomness_matrix = self.gen_randomness()
        self.projection_matrix = self.gen_projection()
        self.offset_matrix = self.gen_offset()

        self.coupling_matrix = self.compute_coupling_matrix(
            self.randomness_matrix, self.variances_matrix, self.projection_matrix, self.offset_matrix
        )

        super().__init__(**kwargs)

    def gen_variances(self):
        return 1.

    def gen_randomnness(self):
        raise NotImplementedError

    def gen_projection(self):
        return 1

    def gen_offset(self):
        return 0

    @staticmethod
    def _compute_coupling_matrix(randomness_matrix, variances_matrix=1., projection_matrix=1., offset_matrix=0.):
        return variances_matrix @ randomness_matrix @ projection_matrix + offset_matrix

    def compute_coupling_matrix(self, randomness_matrix, variances_matrix=1., projection_matrix=1., offset_matrix=0.):
        return self._compute_coupling_matrix(randomness_matrix, variances_matrix, projection_matrix, offset_matrix)
