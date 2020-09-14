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

        if coupling_matrix:
            self.coupling_matrix = coupling_matrix

        elif coupling_strength:
            self.coupling_matrix = get_gaussian_topology(
                self.n_dofs, coupling_strength, self_interaction)

        else:
            raise ValueError(
                "Either `coupling_matrix` or `coupling_strength` must be provided."
            )

    @staticmethod
    def activation(state: np.ndarray) -> np.ndarray:
        return np.tanh(state)

    def take_step(self, t: int, state: np.ndarray) -> np.ndarray:
        return -state + np.dot(self.coupling_matrix, self.activation(state))
