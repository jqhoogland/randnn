import numpy as np

from ..utils import dilute_connectivity
from .base import ContinuousNN


class SparseRandNN(ContinuousNN):
    def __init__(self,
                 sparsity: float=0.,
                 normalize_strength: bool=False,
                 **kwargs) -> None:
        """

        :param sparsity: The proportion of edges per node to set to zero.
        :param normalize_strength: If ``sparsity``, ``normalize_strength`` are set
            then we adjust ``coupling_strength := coupling_strength/ 1 - sparsity``.
            This way we can compare a dense network against a sparse network with
            an equivalent total coupling strength per neuron.
        :param kwargs: see parent class.
        """

        super().__init__(**kwargs)

        self.sparsity = sparsity
        self.sparsity_mask = dilute_connectivity(self.n_dofs, sparsity, False)

        if sparsity and normalize_strength:
            self.coupling_strength /= (1. -sparsity)
            self.coupling_matrix /= (1. -sparsity)

        self._coupling_matrix = self.coupling_matrix
        self.coupling_matrix = np.multiply(self.sparsity_mask, self._coupling_matrix)

    def __repr__(self):
        return "<SparseRandNN coupling_strength:{} sparsity: {} n_dofs:{} timestep:{} seed: {}>".format(
            self.coupling_strength, self.sparsity, self.n_dofs, self.timestep, self.network_seed)
