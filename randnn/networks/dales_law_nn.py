import numpy as np

from randnn.signs import get_dales_law_signs
from .gaussian_nn import GaussianNN


class DalesLawNN(GaussianNN):
    def __init__(self, mu_e: float, frac_excitatory: float = 0.5, **kwargs):
        """
        :param mu_e: the average coupling strength of the excitatory neurons
            ``mu_i``, the average coupling strength of the inhibitory neurons is determined so that the total mean of weights is 0.
        :param frac_excitatory: the fraction of neurons that are excitatory (i.e. outgoing weights are positive)
        :param kwargs: see parent class.
        """
        self.mu_e = mu_e
        self.frac_excitatory = frac_excitatory
        self.frac_inhibitory = 1. - frac_excitatory

        super().__init__(**kwargs)

    def gen_signs(self):
        return get_dales_law_signs(self.n_dofs, self.frac_excitatory, True)

    def gen_weights(self):
        # Let all the signs come from the signs matrix.
        return np.abs(super().gen_weights())
