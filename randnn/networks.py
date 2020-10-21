"""

Contains the various networks I'll be exploring,
i.e. the pairs of neuronal dynamics and network topologies.

Author: Jesse Hoogland
Year: 2020

"""
import numpy as np
from typing import Optional, Union

from .trajectories import DeterministicTrajectory
from .topologies import get_gaussian_topology


class ContinuousNN(DeterministicTrajectory):
    def __init__(self,
                 coupling_strength: Optional[float] = 1.,
                 coupling_matrix: Optional[np.ndarray] = None,
                 self_interaction: bool = False,
                 network_seed: Optional[int] = None,
                 **kwargs) -> None:
        """
        :param coupling_strength: See below.  Either coupling_matrix
            or coupling_strength must be provided, the former taking
            priority.
        :param coupling_matrix: The matrix of couplings between
            neurons.  defaults to a gaussian random coupling matrix
            with variation $g^2/N$, where $N$ is `n_dofs` and $g$ is
            the coupling_stregnth.
        :param self_interaction: This is used if we randomly generatea
            coupling matrix.  It determines whether we do or do not
            allow diagonal elements on the connectivity matrix
        :param network_seed: If we randomly generate a coupling
            matrix, this parameter determines the seed to use for
            np.random.  This is useful if we'd like to compare similar
            networks for different values of the coupling_strength.
            By default, this is left blank, so we do not specify a
            seed.
        :param kwargs: see parent class.
        """

        self.network_seed = network_seed
        super().__init__(**kwargs)

        if not coupling_matrix is None:
            self.coupling_matrix = coupling_matrix
            coupling_strength = np.std(coupling_matrix) * np.sqrt(self.n_dofs)

        elif coupling_strength:
            self.coupling_matrix = get_gaussian_topology(
                self.n_dofs, coupling_strength, self_interaction,
                self.network_seed)
        else:
            raise ValueError("Either `coupling_matrix` or `coupling_strength` must be provided.")

        self.coupling_strength = coupling_strength

    def __repr__(self):
        return "<ContinuousNN coupling_strength:{} n_dofs:{} timestep:{} seed: {}>".format(
            self.coupling_strength, self.n_dofs, self.timestep, self.network_seed)

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
