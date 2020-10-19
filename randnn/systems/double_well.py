"""


Author: Jesse Hoogland
Year: 2020

"""

import numpy as np
from matplotlib import pyplot as plt

from ..trajectories import StochasticTrajectory
from ..integrate import Position

class BrownianMotion(StochasticTrajectory):
    """
    (1d) Brownian motion.

    The equations of motion are given by:
    $$m\ddot x = -\gamma \dot x - V'(x) + \sqrt{2\gamma \beta^{-1}} dW_t,$$
    where
    :param x: is the position of the particle
    :param beta: The thermodynamic beta $\beta^{-1} = k_B T$
    :param gamma: is the damping constant
    :param is_overdamped: is a boolean tag equivalent to taking the limit $m \to 0$.
    - V is the potential (V' denotes its first spatial derivative).
        By default this is 0, though this is overridden in child classes.
    - dW_t is a white noise Wiener process with standard deviation dt.

    At a quick glance, one sees that this is overparametrized. We subsume $m$
    into a redefinition of the other parameters.

    In the overdamped limit, $m\ddot x \ll 1 $, so the E.O.M reduces to:
    $$\gamma \dot x = - V'(x) + \sqrt{2\gamma k_B T} dW_t,$$

    """

    def __init__(self,
                 beta: float = 1.,
                 gamma: float = 1.,
                 is_overdamped: bool = False,
                 **kwargs):
        self.beta = beta
        self.gamma = gamma
        self.is_overdamped = is_overdamped
        super().__init__(**kwargs)

    def __post_init__(self):

        # We run the conditional here so we don't have to go through it every time we integrate a step forward
        if self.is_overdamped:
            self._take_step = lambda t, x: np.array([x[1], -self.gamma * x[1] - self.grad_potential(x[0])])
            self._get_random_step = lambda t, x: np.array([0, np.sqrt(2. * self.gamma / self.beta)])
        else:
            self._take_step = lambda t, x: - self.grad_potential(x) / self.gamma
            self._get_random_step = lambda t,x:  np.sqrt(2. * self.gamma / self.beta) / self.gamma

    @staticmethod
    def grad_potential(x: Position) -> float:
        return 0.

    def take_step(self, t, x: Position) -> Position:
        return self._take_step(t, x)

    def get_random_step(self, t, x) -> Position:
        return self._get_random_step(t, x)


class DoubleWell(BrownianMotion):
    @staticmethod
    def grad_potential(x: Position) -> float:
        return -4 * x * (x**2 - 1)

    def get_boltzmann_dist_exact(self, x_min=-2., n_bins=100, display=True):
        step = -2 * x_min / n_bins
        trajectory = np.arange(x_min, -x_min, step)

        energies = np.square(np.square(trajectory) - 1.)
        boltzmann_weights = np.exp(-energies / (self.temperature))
        boltzmann_weights /= np.sum(boltzmann_weights)

        _ = plt.plot(trajectory, boltzmann_weights)

        if display:
            plt.title("Boltzmann distribution of the double well.")
            plt.xlabel("x")
            plt.ylabel("Probability")
            plt.show()

        return energies, boltzmann_weights

    @property
    def dominant_eigval(self):
        return ((np.sqrt(17) - 1) * np.exp(-1. / self.temperature) / (np.pi * np.sqrt(2)))

    @property
    def dominant_timescale(self):
        return self.timestep / np.log(self.dominant_eigval)
