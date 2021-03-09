from typing import Optional

import numpy as np

from .base_nn import BaseNN, MatrixInit
from ..weights import get_gaussian_weights


class DalesLawNN(BaseNN, MatrixInit):
    def __init__(
            self,
            frac_excitatory: float = 0.5,
            g_excitatory: float = 1.,
            g_inhibitory: Optional[float] = None,
            mu_excitatory: float = 0.,
            balanced=True,
            zero_sum=False,
            n_dofs: int = 100,
            **kwargs
    ):
        """
        ### Fraction excitatory/inhibitory
        :param frac_excitatory ($f_E$ or just $f$): the fraction of neurons that are excitatory
        :param frac_inhibitory ($f_I$): the fraction of neurons that are inhibitory
            - This is fixed by ``frac_excitatory`` ($f_I = 1-f_E$)

        ### Variance of excitatory/inhibitory couplings
        :param g_excitatory ($g_E$): the excitatory population's standard deviation
        :param g_inhibitory ($g_I$): the inhibitory population's standard deviation

        ### Average of excitatory/inhibitory coupling strength
        :param mu_excitatory ($\mu_E$): the average coupling strength of the excitatory neurons
        :param mu_inhibitory ($\mu_I$): the average coupling strength of the inhibitory neurons

        ### Additional constraints
        :param balanced: whether to set the *average* over all edges to zero.
            If this is true, then ``mu_inhibitory`` is fixed by ``mu_excitatory``:
            $$\mu_E f + \mu_I(1-f) = 0$$
        :param zero_sum: whether to enforce *strict* input-output balance (not just on average).
            If this is true, then:
            $$\sum_{j=1}^n(J_{ij} - M_{ij}) = 0$$

        ### Matrices
        :param coupling_matrix ($J$): The final coupling matrix.
            Given by $$J= A \Sigma P + M$$
        :param randomness_matrix ($A$): a normally distributed matrix of zero mean and unit variance,
        :param variances_matrix ($\Sigma$): a diagonal matrix with the variance of neuron $i$ in index $i$.
            - Its first $nf$ elements have value $\mu_E$.
            - The remaining $n(1-f)$ elements have value $\mu_I$.
        :param projection_matrix ($P$): which enforces the ``zero_sum`` constraint
            - If not ``zero_sum``: $P$ is the identity matrix
            - If ``zero_sum``: $P$ is a matrix of all ones with coefficient $1/n$
        :param offset_matrix ($M$): which tracks the offset or average strength of edge $(i, j)$.
            - Its first $nf$ elements have value $g_E$.
            - The remaining $n(1-f)$ elements have value $g_I$.
        """

        assert 0 < frac_excitatory < 1

        if g_inhibitory is None:
            g_inhibitory = g_excitatory

        self.g_excitatory = g_excitatory
        self.g_inhibitory = g_inhibitory

        self.frac_excitatory = frac_excitatory
        self.frac_inhibitory = 1. - frac_excitatory

        self.n_excitatory = round(self.frac_excitatory * n_dofs)
        self.n_inhibitory = round(self.frac_inhibitory * n_dofs)

        self.mu_excitatory = mu_excitatory
        self.mu_inhibitory = -mu_excitatory * frac_excitatory / (1. - frac_excitatory) if balanced else -mu_excitatory

        self.balanced = balanced
        self.zero_sum = zero_sum

        super(BaseNN, self).__init__(n_dofs=n_dofs, **kwargs)
        super(MatrixInit, self).__init__()

    def __repr__(self):
        return "<DalesLawNN n:{} t:{} g_e:{} g_i:{} f_e:{} f_i:{} mu_e:{} mu_i:{} seed:{}>".format(
            self.n_dofs, self.timestep, self.g_excitatory, self.g_inhibitory, self.frac_excitatory,
            self.frac_inhibitory, self.mu_excitatory, self.mu_inhibitory, self.network_seed
        )

    @property
    def get_radius(self):
        return np.sqrt(self.frac_excitatory * self.g_excitatory ** 2 + self.frac_inhibitory * self.g_inhibitory ** 2)

    def gen_variances(self):
        return np.diag([*[self.g_excitatory] * self.n_excitatory, *[self.g_inhibitory] * self.n_inhibitory])

    def gen_randomness(self):
        return get_gaussian_weights(self.n_dofs, 1.)

    def gen_projection(self):
        if self.zero_sum:
            return np.eye((self.n_dofs, self.n_dofs)) - np.ones((self.n_dofs, self.n_dofs)) / self.n_dofs

        return np.eye(self.n_dofs, self.n_dofs)

    def gen_offset(self):
        row = np.array([*([self.mu_excitatory] * round(self.n_excitatory)),
                        *([self.mu_inhibitory] * round(self.n_inhibitory))]).reshape((1, self.n_dofs))

        return np.ones((self.n_dofs, 1)) @ row
