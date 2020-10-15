"""


Author: Jesse Hoogland
Year: 2020

"""

import numpy as np
from matplotlib import pyplot as plt

from ..trajectories import StochasticTrajectory


class DoubleWell(StochasticTrajectory):
    def __init__(self, temperature=0.32, step_size=1e-2, **kwargs):
        """

        Simulates overdamped particle in a double well.

        $$dx_t=-\partial_x V(x_t) dt + (temperature) * dW_t,$$
        where $$V(x)= (x^2-1)^2$$ and $$dW_t$$ is a Weiner process.


        """
        super(DoubleWell, self).__init__(step_size=step_size, **kwargs)

        self.temperature = temperature
        self.beta = 1. / temperature

    def __str__(self):
        return "double_well_t{}.npy".format(np.round(self.temperature, 2))

    def take_step(self, t, x):
        return -4 * x * (x**2 - 1)

    def get_random_step(self, t, x):
        return np.sqrt(2 * self.temperature)

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
        return self.step_size / np.log(self.dominant_eigval)
