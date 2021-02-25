import numpy as np
from randnn.topologies.utils import get_neighbors, dfs_component, get_connected_components

def test_get_neighbors():
    edges = np.array([[1, 1, 1, 1],
                      [0, 1, 0, 1],
                      [0, 0, 1, 1],
                      [1, 1, 0, 0]])

    assert get_neighbors(0, edges) == [0, 1, 2, 3]
    assert get_neighbors(1, edges) == [0, 1, 3]
    assert get_neighbors(2, edges) == [0, 2, 3]
    assert get_neighbors(3, edges) == [0, 1, 2]

def test_dfs_component():
    edges = np.array([[1, 1, 1, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 0]])

    assert dfs_component(0, edges) == [0, 1, 2]
    assert dfs_component(3, edges) == [3]

    edges = np.array([[1, 1, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 1],
                      [0, 0, 1, 1]])

    assert dfs_component(0, edges) == [0, 1]
    assert dfs_component(3, edges) == [3, 2]


def test_get_components():
        edges = np.array([[1, 1, 1, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 0]])

        assert get_connected_components(edges) == {
            3: [3],
            2: [0, 1, 2]
        }

        edges = np.array([[1, 1, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 1],
                          [0, 0, 1, 1]])

        assert get_connected_components(edges) == {
            3: [2, 3],
            1: [0, 1]
        }