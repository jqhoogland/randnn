import numpy as np


def get_fully_connected_edges(n_dofs: int, self_connection: bool = False):
    return np.ones((n_dofs, n_dofs)) - (1 - self_connection) * np.eye(n_dofs)
