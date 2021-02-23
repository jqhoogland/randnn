from typing import Any

import numpy as np
from nptyping import NDArray


def power_law_dist(
        alpha: float,
        max_degree: int,
        min_degree: int
) -> NDArray[(Any, )]:
    """
    An array where index k = p(min_degree + k).

    Bounds are inclusive
    """
    degree_weights = np.power(np.arange(min_degree, max_degree + 1), -alpha)
    return degree_weights / np.sum( degree_weights)


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

def degree_seq_to_edges(degree_seq: NDArray[(Any, )], n_dofs: int) -> NDArray[(Any, Any)]:
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


# ----------------------------------------------------------- -
# TESTING

def test_power_law_dist():
    min_degree = 1
    max_degree = 3

    alpha1 = 0.5
    alpha2 = 1.5
    alpha3 = 2.5
    alpha4 = 3.5

    assert np.isclose(
        power_law_dist(alpha1, max_degree, min_degree),
        np.array([1, 2 ** -0.5, 3 ** -0.5]) / (1 + 2 ** -0.5 + 3 ** -0.5)
    ).all()

    assert np.isclose(
        power_law_dist(alpha2, max_degree, min_degree),
        np.array([1, 2 ** -1.5, 3 ** -1.5]) / (1 + 2 ** -1.5 + 3 ** -1.5)
    ).all()

    assert np.isclose(
        power_law_dist(
            alpha3, max_degree, min_degree),
        np.array([1, 2 ** -2.5, 3 ** -2.5]) / (1 + 2 ** -2.5 + 3 ** -2.5)
    ).all()

    assert np.isclose(
        power_law_dist(alpha4, max_degree, min_degree),
        np.array([1, 2 ** -3.5, 3 ** -3.5]) / (1 + 2 ** -3.5 + 3 ** -3.5)
    ).all()


def test_degree_seq():
    min_degree = 1
    max_degree = 3

    alpha1 = 1.
    alpha2 = 2.

    probs1 = power_law_dist(alpha1, max_degree, min_degree)
    probs2 = power_law_dist(alpha2, max_degree, min_degree)

    # k = 1 2 3
    # alpha = 1: 1 .5 .33... => P(k) = 0.5454..., .2727..., .1818...
    # alpha = 2: 1 .25 .11... => P(k) = 0.7346938776..., 0.1836734694..., 0.08163265306...

    n_dofs1 = 10000

    degrees1 = np.array([*([1] * 5455), *([2] * 2727), *([3] * 1818)])
    degrees2 = np.array([*([1] * 7347), *([2] * 1837), *([3] * 816)])

    # "graphicity"
    # 1 * 5555 + 2 * 2727 + 3 * 1818 = odd
    # 1 * 7347 + 2 * 1837 + 3 * 816 = odd

    degrees1[0] += 1
    degrees2[0] += 1

    assert np.all(get_degree_seq(probs1, n_dofs1, min_degree).shape == degrees1.shape)
    assert np.all(get_degree_seq(probs2, n_dofs1, min_degree).shape == degrees2.shape)

    n_dofs2 = 11

    degrees3 = np.array([*([1] * 6), *([2] * 3), *([3] * 2)])
    degrees4 = np.array([*([1] * 8), *([2] * 2), 3])

    # "graphicity"
    degrees4[0] += 1

    #

    assert np.all(get_degree_seq(probs1, n_dofs2, min_degree) == degrees3)
    assert np.all(get_degree_seq(probs2, n_dofs2, min_degree) == degrees4)


def test_degree_seq_to_edges_matrix():
    n_dofs = 5

    degree_seq1 = [2, 2, 2, 3, 3]
    degree_seq2 = [2, 2, 3, 3, 4]
    degree_seq3 = [2, 3, 3, 4, 4]

    edges_matrix1 = degree_seq_to_edges(degree_seq1, n_dofs)
    edges_matrix2 = degree_seq_to_edges(degree_seq2, n_dofs)
    edges_matrix3 = degree_seq_to_edges(degree_seq3, n_dofs)

    assert np.sum(edges_matrix1[0, :]) == 2
    assert np.sum(edges_matrix1[1, :]) == 2
    assert np.sum(edges_matrix1[2, :]) == 2
    assert np.sum(edges_matrix1[3, :]) == 3
    assert np.sum(edges_matrix1[4, :]) == 3

    assert np.sum(edges_matrix2[0, :]) == 2
    assert np.sum(edges_matrix2[1, :]) == 2
    assert np.sum(edges_matrix2[2, :]) == 3
    assert np.sum(edges_matrix2[3, :]) == 3
    assert np.sum(edges_matrix2[4, :]) == 4

    assert np.sum(edges_matrix3[0, :]) == 2
    assert np.sum(edges_matrix3[1, :]) == 3
    assert np.sum(edges_matrix3[2, :]) == 3
    assert np.sum(edges_matrix3[3, :]) == 4
    assert np.sum(edges_matrix3[4, :]) == 4
