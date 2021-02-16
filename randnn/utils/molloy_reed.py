import logging

import numpy as np
from nptyping import NDArray

def power_law_dist(
        alpha: float,
        max_degree: int,
        min_degree: int
) -> NDArray[(Any, )]:
    """
    An array where index k = p(min_degree + k).
    """
    degree_weights = np.arange(min_degree, max_degree) ** -alpha
    return degree_weights / np.sum( degree_weights)

def get_degree_seq(degree_probs: NDArray[(Any, )], n_dofs: int) -> NDArray[(Any, )]:
    n_nodes_with_degree = (degree_probs * n_dofs).astype(int)

    # Make sure we still have the right number of nodes
    # TODO: Check different ways of resolving this (e.g. random modification)
    n_dofs_rounded = np.sum(n_nodes_with_degree)
    if n_dofs_rounded != n_dofs:
        # We make up the difference with the minimum degree nodes
        # (which likely already have little impact)
        n_nodes_with_degree[0] += n_dofs_rounded - n_dofs

    degree_seq = []
    for i in range(len(degree_probs)):
        degree_seq.extend([i] * n_nodes_with_degree[i])

    degree_seq = np.array(degree_seq)

    # Make sure the sequence is graphic (i.e. the sum of all degrees is even)
    # If it isn't, we add one edge to the largest hub
    if (np.sum(degree_seq) % 2 == 1):
        degree_seq[-1] += 1

    return degree_seq

def degree_seq_to_edges(degree_seq: NDArray[(Any, )], n_dofs: int) -> NDArray[(Any, Any)]:
    # Duplicate each index as many times as its corresponding degree
    # This will produce a list with a length roughly ``2 * n_edges``
    edge_samples = []
    for i in range(n_dofs):
        edge_samples.extend([i] * degree_seq[i])

    # Sample pairs of digits from this list to will form a new edge.
    # then remove the pair from the list.
    # Remember edges are directed.

    edges_matrix = np.zeros((n_dofs, n_dofs))

    # This may leave a few extra edges
    # (if at the end one node is left repeated but there are no other nodes to pair to).
    while len(edge_samples) and not len(set(edge_samples)) == 1:
        [i, j] = np.random.choice(len(edge_samples), size=2, replace=False)

        # No self-connections & no doubled edges.
        if (edge_samples[i] == edge_samples[j] or edges_matrix[i, j] != 0):
            continue

        edges_matrix[i, j] = 1

        del(edge_samples[i])
        del(edge_samples[j])

    if len(edge_samples):
        logging.log(f"Node {edge_samples[0]} is below degree by {len(edge_samples)/2}")

    return edges_matrix

def molloy_reed(
        alpha: float,
        max_degree: int,
        min_degree: int,
        n_dofs: int
):
    """
    Based on the Molloy-Reed approach via Coscia 2021

    :param alpha: the exponent in the power law.
    :param max_degree: the maximum allowed degree (defaults to ``n_dofs-1``, i.e., all other nodes)
    :param min_degree: the minimum allowed degree. Must be >= 1 (defaults to 1)
    :param n_dofs: the number of degrees of freedom
    """

    # (1) compute probabilities (an array where index k = p(min_degree + k))
    degree_probs = power_law_dist(alpha, max_degree, min_degree)

    # (2) convert to degree sequence ([k_min...k_min, k_min + 1...k_min + 1, ... , k_max...k_max  ])
    # where each value appears n(k) times. This will have a length = ``n_dofs``
    degree_seq = get_degree_seq(degree_probs, n_dofs)

    # (3) convert to edges matrix
    edges_matrix = degree_seq_to_edges(degree_seq, n_dofs)

    return edges_matrix
