# ------------------------------------------------------------
# TESTING
import numpy as np
from randnn.weights import get_gaussian_weights

def test_gaussian_weights():
    for g in np.arange(0.5, 20, 0.5):
        for n in range(50, 1000, 100):
            g_squiggle = g ** 2 / n
            coupling_matrix = get_gaussian_weights(n, g)

            assert np.allclose(
                np.var(coupling_matrix),
                g_squiggle,
                rtol=1e-4
            ), "GaussianNN not initializing coupling matrix with correct weight dist."
