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
    def __init__(
            self,
            init_state: Optional[np.ndarray[np.float64]] = None,
            coupling_matrix=Union[np.ndarray[np.float64], float]=1.,
            n_dofs: Optional[int] = 100,
            **kwargs) -> None:
        """
        :param init_state: the state to initialize the neurons with.
            defaults to a state of `n_dofs` neurons drawn randomly from the uniform distribution.
            if left blank, then `n_dofs` must be specified.
        :param coupling_matrix: the matrix of couplings between neurons.
            if this parameter is of type `float`, then the coupling matrix defaults to a gaussian random
            coupling matrix with variation $g^2/N$, where $N$ is `n_dofs`.
            defaults to 1.0
        :param n_dofs: the number of dofs.
            if `init_state` is of type `int`, this must be specified,
            else `n_dofs` is overwritten by the size of `init_state`
        :param kwargs: see parent class.
        """

        if type(init_state) == "int":
            assert not n_dofs is None
            init_state = np.random.uniform(size=n_dofs)
        else:
            n_dofs = init_state.size

        if type(coupling_matrix) == "float":
            coupling_matrix = get_gaussian_topology(
                n_dofs, coupling_matrix, kwargs.get("self_interaction", false))

        self.init_state = init_state
        self.coupling_matrix = coupling_matrix
        super().__init__(n_dofs=n_dofs, **kwargs)

    @staticmethod
    def activation(state: np.ndarray) -> np.ndarray:
        return np.tanh(state)

    def take_step(self, t: int, state: np.ndarray) -> np.ndarray:
        return -state + np.dot(self.coupling_matrix, self.activation(state))
