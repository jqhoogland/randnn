import numpy as np

from randnn.topologies import get_fully_connected_edges, dilute_connectivity
# ----------------------------------------------------------- -
# FULLY-CONNECTED
# ----------------------------------------------------------- -

def test_fully_connected_edges():
    edge_matrix = get_fully_connected_edges(10000, False)
    assert np.allclose(np.diag(edge_matrix), np.zeros(10000))
    assert np.allclose(edge_matrix[edge_matrix != 0], np.ones((10000,10000))[edge_matrix != 0])

# ----------------------------------------------------------- -
# UNIFORM DILUTION
# ----------------------------------------------------------- -

def test_dilute_connectivity():
    # sparsity must be in [0, 1]

    try:
        dilute_connectivity(100, 1.1)
        assert False
    except AssertionError:
        assert True

    try:
        dilute_connectivity(100, -0.1)
        assert False
    except AssertionError:
        assert True

    for s in np.arange(0, 1, 0.2):
        for n in range(100, 1000, 200):
            mask_1 = dilute_connectivity(n, s, True)
            assert np.sum(mask_1) == round((n ** 2) * (1 - s))

            mask_2 = dilute_connectivity(n, s, False)
            assert np.sum(mask_2) == round(n * (n - 1) * (1 - s))
            assert np.all(np.diagonal(mask_2) == 0)


