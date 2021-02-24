import math

import numpy as np

from randnn.networks.weights.gaussian import GaussianNN
from .topology import dilute_connectivity


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

        super().__init__(**kwargs)

        self.sparsity = sparsity
        self.edges_matrix = dilute_connectivity(self.n_dofs, sparsity, False)

        self.update_coupling_matrix(self.weights_matrix, self.edges_matrix)

        if normalize_strength:
            # If `normalize_strength`, the edge weights over all pairs of nodes
            # (edge or no edge) should have a stdev = g / \sqrt{N}
            self.weights_matrix *= self.coupling_strength / (np.sqrt(self.n_dofs) * np.std(self.coupling_matrix))
        else:
            # Otherwise, the edge weights over all pairs of nodes
            # should have a stdev = (1 - s ) g / \sqrt{N}
            self.weights_matrix *= (
                    self.coupling_strength * (1 - sparsity) / (np.sqrt(self.n_dofs) * np.std(self.coupling_matrix))
            )

        self.update_coupling_matrix(self.weights_matrix, self.edges_matrix)
        self.coupling_strength = np.std(self.coupling_matrix) * np.sqrt(self.n_dofs)

    def __repr__(self):
        return "<SparseRandNN coupling_strength:{} sparsity: {} n_dofs:{} timestep:{} seed: {}>".format(
            self.coupling_strength, self.sparsity, self.n_dofs, self.timestep, self.network_seed
        )


# ------------------------------------------------------------
# TESTING

# TODO: Get non-normalized tests to work
def test_connectivity_matrices():
    for s in np.arange(0, 1.0, 0.1):
        for g in np.arange(0.5, 20, 2.5):
            for n in range(100, 500, 200):
                g_squiggle = g ** 2 / n
                nn = SparseRandNN(sparsity=s, coupling_strength=g, n_dofs=n)

                assert np.allclose(
                    np.sum(nn.edges_matrix) / (n * (n - 1)),
                    1 - s,
                    rtol=1e-4
                ), f"SparseRandNN s={s} g={g} n={n} not initializing edges correctly."

                assert np.allclose(
                    np.var(nn.weights_matrix),
                    g_squiggle * (1 - s) ** 2,
                    rtol=1e-1
                ), f"SparseRandNN s={s} g={g} n={n} not initializing coupling matrix correctly."

                assert np.all(
                    (nn.coupling_matrix == 0) == (nn.edges_matrix == 0)
                ), f"SparseRandNN s={s} g={g} n={n} not initializing coupling matrix correctly."

                assert np.allclose(
                    np.var(nn.coupling_matrix),
                    g_squiggle * (1 - s) ** 2,
                    rtol=1e-4
                ), f"SparseRandNN s={s} g={g} n={n} not initializing coupling matrix correctly."

                assert np.allclose(
                    nn.coupling_strength,
                    g * (1 - s),
                    rtol=1e-4
                ), f"SparseRandNN s={s} g={g} n={n} not computing coupling_strength correctly."


def test_normalize_strength():
    for s in np.arange(0, 1.0, 0.1):
        for g in np.arange(0.5, 20, 2.5):
            for n in range(100, 500, 200):
                g_squiggle = g ** 2 / n
                nn = SparseRandNN(sparsity=s, coupling_strength=g, n_dofs=n, normalize_strength=True)

                assert np.allclose(
                    np.sum(nn.edges_matrix) / (n * (n - 1)),
                    1 - s,
                    rtol=1e-4
                ), f"SparseRandNN s={s} g={g} n={n} not initializing edges correctly."

                assert np.all(
                    (nn.coupling_matrix == 0) == (nn.edges_matrix == 0)
                ), f"SparseRandNN s={s} g={g} n={n} not initializing coupling matrix correctly."

                assert np.allclose(
                    np.var(nn.coupling_matrix),
                    g_squiggle,
                    rtol=1e-4
                ), f"SparseRandNN s={s} g={g} n={n} not initializing coupling matrix correctly."

                assert math.isclose(
                    nn.coupling_strength,
                    g,
                    rel_tol=1e-4
                ), f"SparseRandNN s={s} g={g} n={n} not computing coupling_strength correctly."
