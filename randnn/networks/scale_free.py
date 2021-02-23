"""

Contains the various networks I'll be exploring,
i.e. the pairs of neuronal dynamics and network topologies.

Author: Jesse Hoogland
Year: 2020

"""
from typing import Optional

import numpy as np

from .gaussian import GaussianNN
from ..utils import get_scale_free_edge_matrix


class ScaleFreeNN(GaussianNN):
    def __init__(self,
                 alpha: float = -2.,
                 max_degree: Optional[int] = None,
                 min_degree: Optional[int] = None,
                 normalize_strength: bool = False,
                 **kwargs) -> None:
        """
        A network with a power law distribution:
        $$P(k)\sim ck^{-\alpha}$$

        :param alpha: the exponent in the power law.
        :param max_degree: the maximum allowed degree (defaults to ``n_dofs-1``,
            i.e., all other nodes)
        :param min_degree: the minimum allowed degree. Must be >= 1 (defaults to 1)
        :param kwargs: see parent class.
        """
        super().__init__(**kwargs)

        if max_degree is None:
            max_degree = self.n_dofs - 1

        if min_degree is None:
            min_degree = 1

        assert alpha >= 0, "``alpha`` must be positive"
        assert 0 < min_degree < max_degree < self.n_dofs

        self.alpha = alpha
        self.max_degree = max_degree
        self.min_degree = min_degree

        self.edges_matrix = get_scale_free_edge_matrix(
            alpha, max_degree, min_degree, self.n_dofs
        )
        self.coupling_matrix = np.multiply(self.edges_matrix, self.weights_matrix)

        if normalize_strength:
            true_strength = np.std(self.coupling_matrix)

            self.coupling_matrix *= self.coupling_strength / true_strength

    def __repr__(self):
        return f"<ScaleFreeNN coupling_strength:{self.coupling_strength} std:{np.std(self.coupling_matrix)} alpha:{self.alpha} min_degree:{self.min_degree} max_degree:{self.max_degree} n_dofs:{self.n_dofs} timestep:{self.timestep} seed: {self.network_seed} >"

    @property
    def avg_degree(self):
        return np.sum(np.where(self.coupling_matrix > 0, 1, 0))



# ------------------------------------------------------------
# TESTING
