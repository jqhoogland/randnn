from typing import Any

import numpy as np
from nptyping import NDArray


def power_law_dist(
        alpha: float,
        max_degree: int,
        min_degree: int
) -> NDArray[(Any,)]:
    """
    An array where index k = p(min_degree + k).

    Bounds are inclusive
    """
    degree_weights = np.power(np.arange(min_degree, max_degree + 1), -alpha)
    return degree_weights / np.sum(degree_weights)


def get_degree_seq(degree_probs: NDArray[(Any,)], n_dofs: int, min_degree: int = 1) -> NDArray[(Any,)]:
    n_nodes_with_degree = np.round(degree_probs * n_dofs).astype(int)

    degree_seq = []
    for i in range(len(degree_probs)):
        degree_seq.extend([min_degree + i] * n_nodes_with_degree[i])

    degree_seq = np.array(degree_seq)

    # Make sure we still have the right number of nodes
    # TODO: Check different ways of resolving this (e.g. random modification)
    n_dofs_rounded = len(degree_seq)
    if n_dofs_rounded < n_dofs:
        degree_seq = np.concatenate(
            [np.array([min_degree] * (n_dofs - n_dofs_rounded)), degree_seq]
        )
    elif n_dofs_rounded > n_dofs:
        degree_seq = degree_seq[(n_dofs_rounded - n_dofs):]

    # Make sure the sequence is graphic (i.e. the sum of all degrees is even)
    # If it isn't, we add one edge to one of th min_degree units
    if (np.sum(degree_seq) % 2 == 1):
        degree_seq[0] += 1

    return degree_seq


def degree_seq_to_edges(degree_seq: NDArray[(Any,)], n_dofs: int) -> NDArray[(Any, Any)]:
    edges_matrix = np.zeros((n_dofs, n_dofs))

    for i in range(n_dofs):
        n_neighbors = degree_seq[i]

        # No self-connections
        other_nodes = [j for j in range(n_dofs) if j != i]

        # No replacement => No doubled edges
        neighbors = np.random.choice(other_nodes, size=n_neighbors, replace=False)

        for neighbor in neighbors:
            edges_matrix[i, neighbors] = 1

    return edges_matrix


def get_scale_free_edge_matrix(
        alpha: float,
        max_degree: int,
        min_degree: int,
        n_dofs: int
):
    """
    Generate a scale free degree distribution, then turn this into a binary edge matrix
    (i.e., an unweighted adjacency matrix)

    :param alpha: the exponent in the power law.
    :param max_degree: the maximum allowed degree (defaults to ``n_dofs-1``, i.e., all other nodes)
    :param min_degree: the minimum allowed degree. Must be >= 1 (defaults to 1)
    :param n_dofs: the number of degrees of freedom
    """

    # (1) compute probabilities (an array where index k = p(min_degree + k))
    degree_probs = power_law_dist(alpha, max_degree, min_degree)

    # (2) convert to degree sequence ([k_min...k_min, k_min + 1...k_min + 1, ... , k_max...k_max  ])
    # where each value appears n(k) times. This will have a length = ``n_dofs``
    degree_seq = get_degree_seq(degree_probs, n_dofs, min_degree)

    # (3) convert to edges matrix
    edge_matrix = degree_seq_to_edges(degree_seq, n_dofs)

    return edge_matrix
