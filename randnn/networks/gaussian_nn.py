import numpy as np

from randnn.weights import get_gaussian_weights
from .base_nn import BaseNN, ElementWiseInit


class GaussianNN(BaseNN, ElementWiseInit):
    def __init__(self,
                 coupling_strength: float = 1.,
                 **kwargs) -> None:
        """
        :param coupling_strength: see ``get_gaussian_weights``
        :param kwargs: see parent class.
        """
        self.coupling_strength = coupling_strength

        super().__init__(**kwargs)

    def __repr__(self) -> str:
        return "<GaussianNN coupling_strength:{} n_dofs:{} timestep:{} seed: {}>".format(
            self.coupling_strength, self.n_dofs, self.timestep, self.network_seed)

    @property
    def get_radius(self):
        return self.coupling_strength

    @staticmethod
    def _gen_weights(n_dofs: int, g: float) -> np.ndarray:
        return get_gaussian_weights(n_dofs, g)

    def gen_weights(self) -> np.ndarray:
        return self._gen_weights(self.n_dofs, self.coupling_strength)
