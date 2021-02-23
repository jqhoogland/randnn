"""

Contains the various networks I'll be exploring,
i.e. the pairs of neuronal dynamics and network topologies.

Author: Jesse Hoogland
Year: 2020

"""
from typing import Optional

import numpy as np

from pynamics.trajectories import DeterministicTrajectory


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
        np.random.seed(network_seed)

        self.update_coupling_matrix(np.eye(kwargs.get("n_dofs", 100)))

    def __repr__(self):
        return "<BaseNN n_dofs:{} timestep:{} seed: {}>".format(
            self.coupling_strength, self.n_dofs, self.timestep, self.network_seed)

    def update_coupling_matrix(self, weights_matrix, edges_matrix=1):
        self.coupling_matrix = np.multiply(weights_matrix, edges_matrix)

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


# ------------------------------------------------------------
# TESTING

def test_coupling_matrix():
    for n in range(10, 100, 30):
        assert np.allclose(
            BaseNN(n_dofs=n).coupling_matrix,
            np.eye(n)
        ), "BaseNN not initializing coupling matrix with correct # dofs "


def test_jacobian():
    for n in range(10, 100, 30):
        assert np.allclose(
            BaseNN(n_dofs=n, timestep=0.3).jacobian(np.zeros(n)),
            (-0.4 * np.eye(n))
        ), "BaseNN not correctly calculating jacobian when state is 0"


def test_jacobian_saturation():
    for n in range(10, 100, 30):
        assert np.allclose(
            BaseNN(n_dofs=n, timestep=0.3).jacobian(np.ones(n) * 1e10),
            -np.eye(n) * 0.7
        ), "BaseNN not correctly calculating jacobian when state is uniformly large"
