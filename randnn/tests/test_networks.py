import numpy as np

from randnn.networks import BaseNN, SparseRandNN


# ------------------------------------------------------------
# GAUSSIAN NETS

class NN1(BaseNN):
    def gen_weights(self):
        return np.eye(self.n_dofs)

    def gen_edges(self):
        return np.ones((self.n_dofs, self.n_dofs))

class NN2(BaseNN):
    def gen_weights(self):
        return np.eye(self.n_dofs, k=1) + np.eye(self.n_dofs, k=-1)

    def gen_signs(self):
        return np.ones((self.n_dofs, self.n_dofs)) - 2 * np.eye(self.n_dofs, k=-1)

def test_coupling_matrix():
    for n in range(10, 100, 30):
        assert np.allclose(
            NN1(n_dofs=n).coupling_matrix,
            np.eye(n)
        )

        assert np.allclose(
            NN2(n_dofs=n).coupling_matrix,
            np.eye(n, k=1) - np.eye(n, k=-1)
        )


def test_jacobian():
    for n in range(10, 100, 30):
        assert np.allclose(
            NN1(n_dofs=n, timestep=0.3).jacobian(np.zeros(n)),
            (-0.4 * np.eye(n))
        )


def test_jacobian_saturation():
    for n in range(10, 100, 30):
        assert np.allclose(
            NN1(n_dofs=n, timestep=0.3).jacobian(np.ones(n) * 1e10),
            -np.eye(n) * 0.7
        )


# ------------------------------------------------------------
# SPARSE NETS

# TODO: Get non-normalized tests to work
def test_connectivity_matrices():
    for s in np.arange(0, 1.0, 0.1):
        for g in np.arange(0.5, 20, 2.5):
            for n in range(100, 500, 200):
                g_squiggle = g ** 2 / n
                nn = SparseRandNN(sparsity=s, coupling_strength=g, n_dofs=n)

                non_diagonal_idxs = np.where(np.ones((n, n)) - np.eye(n))

                assert np.allclose(
                    np.sum(nn.edges_matrix) / (n * (n - 1)),
                    1 - s,
                    rtol=1e-4
                ), f"SparseRandNN s={s} g={g} n={n} not diluting the correct # of edges."

                assert np.allclose(
                    np.var(nn.coupling_matrix[non_diagonal_idxs]),
                    g_squiggle * (1 - s) ** 2,
                    rtol=1e-4
                ), f"SparseRandNN s={s} g={g} n={n} not reweighting edges correctly."

def test_normalize_strength():
    for s in np.arange(0, 1.0, 0.1):
        for g in np.arange(0.5, 20, 2.5):
            for n in range(100, 500, 200):
                g_squiggle = g ** 2 / n
                nn = SparseRandNN(sparsity=s, coupling_strength=g, n_dofs=n, normalize_strength=True)

                non_diagonal_idxs = np.where(np.ones((n, n)) - np.eye(n))

                assert np.allclose(
                    np.sum(nn.edges_matrix) / (n * (n - 1)),
                    1 - s,
                    rtol=1e-4
                ), f"SparseRandNN s={s} g={g} n={n} not diluting the correct # of edges."

                assert np.allclose(
                    np.var(nn.coupling_matrix[non_diagonal_idxs]),
                    g_squiggle,
                    rtol=1e-4
                ), f"SparseRandNN s={s} g={g} n={n} not reweighting edges correctly."
