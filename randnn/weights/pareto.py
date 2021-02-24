import numpy as np


def get_pareto_weights(n_dofs: int, alpha: float, coupling_strength: float) -> np.ndarray:
    """
    $J_{ij}$ is drawn from $f(x; 1/\beta) = \exp(-x / \beta) / \beta$

    :param n_dofs: the number of nodes in the network
    :param coupling_strength: the scale parameter (equiv. the inverse rate).
        the final couplings are drawn from an exponential distribution with variation $g^2/N$, where $g$ is the
        coupling strength and $N$ is the number of nodes.
    """
    strength_normalized = (coupling_strength / np.sqrt(n_dofs))
    coupling_matrix = np.random.pareto(size=(n_dofs, n_dofs), a=alpha)
    coupling_matrix *= np.random.choice([1, -1], size=(n_dofs, n_dofs))  # random sign for each node

    coupling_matrix *= (strength_normalized / np.std(coupling_strength))

    return coupling_matrix
