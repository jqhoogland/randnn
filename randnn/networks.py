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
                 self_interaction: Optional[bool] = False,
                 **kwargs) -> None:
        """
        :param coupling_strength: see below. Either coupling_matrix or coupling_strenght must be provided, the former taking priority.
        :param coupling_matrix: the matrix of couplings between neurons.
            defaults to a gaussian random coupling matrix with variation $g^2/N$, where $N$ is `n_dofs` and $g$ is the coupling_stregnth.

        :param kwargs: see parent class.
        """

        super().__init__(**kwargs)

        if not coupling_matrix is None:
            self.coupling_matrix = coupling_matrix
            coupling_strength = np.std(coupling_matrix) * np.sqrt(self.n_dofs)

        elif coupling_strength:
            self.coupling_matrix = get_gaussian_topology(
                self.n_dofs, coupling_strength, self_interaction)

        else:
            raise ValueError(
                "Either `coupling_matrix` or `coupling_strength` must be provided."
            )

        self.timestep = kwargs.get("step_size", 0.01)

        self.coupling_strength = coupling_strength

    def __repr__(self):
        return "<ContinuousNN coupling_strength:{} n_dofs:{} timestep:{}>".format(
            self.coupling_strength, self.n_dofs, self.timestep)

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
