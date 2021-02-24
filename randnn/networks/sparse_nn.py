import numpy as np

from randnn.topologies import dilute_connectivity
from .gaussian_nn import GaussianNN


class SparseRandNN(GaussianNN):
    def __init__(self,
                 sparsity: float = 0.,
                 normalize_strength: bool = False,
                 **kwargs) -> None:
        """

        TODO: Do I want to include the diagonal of the connectivity matrix when normalizing?
        :param sparsity: The proportion of edges per node to set to zero.
        :param normalize_strength: If ``sparsity``, ``normalize_strength`` are set
            then we adjust ``coupling_strength := coupling_strength/ 1 - sparsity``.
            This way we can compare a dense network against a sparse network with
            an equivalent total coupling strength per neuron.
        :param kwargs: see parent class.
        """
        self.sparsity = sparsity
        self.normalize_strength = normalize_strength

        super().__init__(**kwargs)

    def __repr__(self):
        return "<SparseRandNN coupling_strength:{} sparsity: {} n_dofs:{} timestep:{} seed: {} normalize:{}>".format(
            self.coupling_strength, self.sparsity, self.n_dofs, self.timestep, self.network_seed,
            self.normalize_strength
        )

    def gen_edges(self):
        return dilute_connectivity(self.n_dofs, self.sparsity, False)

    def compute_coupling_matrix(self, weights_matrix, edges_matrix=1, signs_matrix=1):
        coupling_matrix_unnormalized = self._compute_coupling_matrix(weights_matrix, edges_matrix, signs_matrix)

        if self.normalize_strength:
            # If `normalize_strength`, the edge weights over all pairs of nodes
            # (edge or no edge) should have a stdev = g / \sqrt{N}
            self.weights_matrix *= self.coupling_strength / (
                    np.sqrt(self.n_dofs) * np.std(coupling_matrix_unnormalized[self.non_diagonal_idxs]))
        else:
            # Otherwise, the edge weights over all pairs of nodes
            # should have a stdev = (1 - s ) g / \sqrt{N}
            self.weights_matrix *= (
                    self.coupling_strength * (1 - self.sparsity) / (
                    np.sqrt(self.n_dofs) * np.std(coupling_matrix_unnormalized[self.non_diagonal_idxs]))
            )

        coupling_matrix = self._compute_coupling_matrix(self.weights_matrix, self.edges_matrix, self.signs_matrix)

        return coupling_matrix
