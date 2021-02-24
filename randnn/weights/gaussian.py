import numpy as np


def get_gaussian_weights(
        n_dofs: int, coupling_strength: float,
) -> np.ndarray:
    """
    :param n_dofs: the number of nodes in the network
    :param coupling_strength: the final couplings are drawn from a
        normal distribution with variation $g^2/N$, where $g$ is the
        coupling strength and $N$ is the number of nodes.
    """
    strength_normalized = (coupling_strength / np.sqrt(n_dofs))
    unit_matrix = np.random.normal(size=(n_dofs, n_dofs))

    coupling_matrix = (strength_normalized * unit_matrix / np.std(unit_matrix))

    return coupling_matrix
