import numpy as np

from randnn.networks import BaseNN, SparseRandNN


# ------------------------------------------------------------
# GAUSSIAN NETS

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


# ------------------------------------------------------------
# SPARSE NETS

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

                assert np.allclose(
                    nn.coupling_strength,
                    g,
                    rtol=1e-4
                ), f"SparseRandNN s={s} g={g} n={n} not computing coupling_strength correctly."
