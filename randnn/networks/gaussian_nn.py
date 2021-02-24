from randnn.weights import get_gaussian_weights
from .base_nn import BaseNN


class GaussianNN(BaseNN):
    def __init__(self,
                 coupling_strength: float = 1.,
                 **kwargs) -> None:
        """
        :param coupling_strength: see ``get_gaussian_weights``
        :param kwargs: see parent class.
        """
        self.coupling_strength = coupling_strength

        super().__init__(**kwargs)

    def __repr__(self):
        return "<GaussianNN coupling_strength:{} n_dofs:{} timestep:{} seed: {}>".format(
            self.coupling_strength, self.n_dofs, self.timestep, self.network_seed)

    def gen_weights(self):
        return get_gaussian_weights(self.n_dofs, self.coupling_strength)
