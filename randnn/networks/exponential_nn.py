from .base_nn import BaseNN

class ExponentialNN(BaseNN):
    def __init__(self,
                 coupling_strength: float = 1.,
                 **kwargs) -> None:
        """
        :param coupling_strength: see ``get_exponential_weights``
        :param kwargs: see parent class.
        """
        self.coupling_strength = coupling_strength

        super().__init__(**kwargs)

    def __repr__(self):
        return f"<ExponentialNN coupling_strength:{self.coupling_strength} n_dofs:{self.n_dofs} timestep:{self.timestep} seed: {self.network_seed}>"

    def gen_weights(self):
        return get_exponential_weights(self.n_dofs, self.coupling_strength)

