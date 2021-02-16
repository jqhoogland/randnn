from typing import Optional

import numpy as np



def dilute_connectivity(n_dofs: int, sparsity: Optional[float]=None, self_interaction: bool=False):
    """
    :param n_dofs: the dimension of the connectivity matrix.
    :param sparsity: the sparsity coefficient.
    :param self_interaction: whether to allow on-diagonal elements. TODO
    """
    if sparsity is None:
        return 1.

    assert 0 <= sparsity <= 1., f"``sparsity`` must be greater than 0 or less than 1, is {sparsity}"

    sparsity_mask = np.ones([n_dofs, n_dofs])
    n_edges = int(n_dofs * (n_dofs + self_interaction - 1))
    n_edges_deleted = int(n_edges * sparsity)

    if self_interaction is False:
        sparsity_mask[np.diag_indices_from(sparsity_mask)] = 0

    indices = []
    for i in range(n_dofs):
        for j in range(n_dofs):
            if i != j or self_interaction:
                indices.append([i, j])

    indices = np.array(indices)
    assert indices.shape[0] == n_edges

    diluted_indices = indices[np.random.choice(n_edges, size=n_edges_deleted, replace=False)]

    for (i, j) in diluted_indices:
        # There's definitely a cleverer array slicing way to do this
        sparsity_mask[i, j] = 0

    return sparsity_mask

def test_dilute_connectivity():
    n_dofs = 100

    # sparsity must be in [0, 1]
    try:
        mask_1 = dilute_connectivity(n_dofs, 1.1)
        assert False
    except AssertionError:
        assert True

    # sparsity = 0 preserves all connections
    mask_2 = dilute_connectivity(n_dofs, 0., True)
    assert np.sum(mask_2) == n_dofs ** 2

    mask_3 = dilute_connectivity(n_dofs, 0., False)
    assert np.sum(mask_3) == n_dofs * (n_dofs - 1)
    assert np.all(np.diagonal(mask_3) == 0)

    # sparsity = 1 deletes all connections
    mask_4 = dilute_connectivity(n_dofs, 1., False)
    assert np.sum(mask_4) == 0

    # sparsity = 0.4 deletes 40% of the connections
    mask_5 = dilute_connectivity(n_dofs, 0.4, True)
    assert np.sum(mask_5) == n_dofs ** 2 * (1 - 0.4)

    mask_6 = dilute_connectivity(n_dofs, 0.4, False)
    assert np.sum(mask_6) == n_dofs * (n_dofs - 1) * (1 - 0.4)
    assert np.all(np.diagonal(mask_6) == 0)
