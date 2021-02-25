import numpy as np

from randnn.topologies.utils import get_connected_components
from randnn.topologies.scale_free import get_degree_seq, degree_seq_to_edges, power_law_dist

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

# TODO: Add support for in and out degree dist.


def test_connectedness():
    # TODO: This one passes depending on your seed.
    for n  in range(100, 1000, 100):
        min_degree = 1
        max_degree = n - 1

        for alpha in [0, 0.5, 1., 1.5, 2., 2.5,  3., 3.5]:
            dist = power_law_dist(alpha, max_degree, min_degree)
            degrees = get_degree_seq(dist, n, min_degree)
            edges = degree_seq_to_edges(degrees, n)

            components = get_connected_components(edges)
            component_sizes = { root: len(component) for (root, component) in components.items()}

            assert len(components.keys()) == 1, f"More than one component : {component_sizes} with alpha={alpha} n={n}"

    # TODO: Add in a check when generating a matrix that it's connected. If not, generate a new matrix.