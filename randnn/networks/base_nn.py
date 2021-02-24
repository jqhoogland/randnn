"""

Contains the various networks I'll be exploring,
i.e. the pairs of neuronal dynamics and network topologies.

Author: Jesse Hoogland
Year: 2020

"""
from typing import Optional

import numpy as np

from pynamics.trajectories import DeterministicTrajectory
from ..topology import get_fully_connected_edges


class BaseNN(DeterministicTrajectory):
    def __init__(self,
                 network_seed: Optional[int] = None,
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
        super().__init__(**kwargs)
        self.network_seed = network_seed

        if network_seed:
            np.random.seed(network_seed)

        # Binary matrix with (i, j) = 1 if there is an edge from i to j, 0 otherwise.
        self.edges_matrix = self.gen_edges()

        # Weights matrix generates a weight for *every* possible pair of nodes
        self.weights_matrix = self.gen_weights()

        # Signs generates a sign multiplier for every possible edge.
        # By default this is a matrix of ones (i.e. do not change the edge weights).
        # This becomes important for Dale's law & one-sided edge-weight distributions.
        self.signs_matrix = self.gen_signs()

        # Final adjacency matrix where (i, j) = weight in weights matrix iff (i, j) == 1 in edges matrix
        # This is the element-wise product of the edges and weights matrices.
        self.coupling_matrix = self.compute_coupling_matrix(self.weights_matrix, self.edges_matrix, self.signs_matrix)

    def __repr__(self):
        return "<BaseNN n_dofs:{} timestep:{} seed: {}>".format(
            self.coupling_strength, self.n_dofs, self.timestep, self.network_seed)

    def gen_edges(self):
        return get_fully_connected_edges(self.n_dofs, False)

    def gen_weights(self):
        raise NotImplementedError

    def gen_signs(self):
        return np.ones((self.n_dofs, self.n_dofs))

    @staticmethod
    def _compute_coupling_matrix(weights_matrix, edges_matrix=1, signs_matrix=1):
        return np.multiply(np.multiply(weights_matrix, edges_matrix), signs_matrix)

    def compute_coupling_matrix(self, weights_matrix, edges_matrix=1, signs_matrix=1):
        return self._compute_coupling_matrix(weights_matrix, edges_matrix, signs_matrix)

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

