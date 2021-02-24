# ------------------------------------------------------------
# TESTING


def test_get_gaussian_topology():
    coupling_matrix = get_gaussian_weights(10000, 10., self_interaction=True)
    assert np.isclose(np.std(coupling_matrix), 0.1, rtol=0.001)


def test_coupling_matrix():
    for g in np.arange(0.5, 20, 0.5):
        for n in range(50, 1000, 100):
            g_squiggle = g ** 2 / n
            coupling_matrix = get_gaussian_weights(n, g).coupling_matrix

            assert np.allclose(
                np.var(coupling_matrix),
                g_squiggle,
                rtol=1e-3
            ), "GaussianNN not initializing coupling matrix with correct weight dist."
