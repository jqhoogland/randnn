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
    n_edges_deleted = round(n_edges * sparsity)

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

