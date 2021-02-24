from typing import Optional, List

import numpy as np


def get_neighbors(node_idx: int, edges: np.ndarray):
    # This is all out-node neighbors
    in_neighbors = np.where(edges[:, node_idx].reshape((-1,)) != 0)[0].tolist()
    out_neighbors = np.where(edges[node_idx, :].reshape((-1,)) != 0)[0].tolist()

    # TODO: Distinguish in-nodes & out-nodes
    return list(set(
        [
            *in_neighbors, *out_neighbors
        ]
    ))


def dfs_component(root: int, edges: np.ndarray, visited: Optional[List[int]] = None) -> List[int]:
    # Get all nodes connected to root by a dfs.
    new_nodes = [root]

    if visited is None:
        visited = [root]

    # Determine the neighbors
    neighbors = get_neighbors(root, edges)

    # Skip any already ``visited`` nodes
    unvisited_neighbors = [neighbor for neighbor in neighbors if neighbor not in visited]

    # Add these new nodes to ``visited``
    visited.extend(unvisited_neighbors)

    # Add all of the nodes connected to these nodes to ``visited``
    # Recursively call the dfs.
    for neighbor in unvisited_neighbors:
        children = dfs_component(neighbor, edges, visited=visited)
        visited.extend(children)
        new_nodes.extend(children)

    return new_nodes


def get_connected_components(edges: np.ndarray) -> dict:
    unvisited = list(range(edges.shape[0]))
    visited = []
    components = {}

    while len(unvisited) > 0:
        # Visit the next unvisited node
        i = unvisited.pop()

        # Use a DFS to get all of the nodes in its component
        component = dfs_component(i, edges)
        component.sort()

        components[i] = component

        # Add these nodes to ``visited``
        visited.extend(components[i])

        # Remove them from ``unvisited``
        for node in components[i]:
            if node != i:
                unvisited.remove(node)

    return components
