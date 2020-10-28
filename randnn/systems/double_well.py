"""

Author: Jesse Hoogland
Year: 2020

"""
from typing import Union, Any, Callable

from nptyping import NDArray
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from ..trajectories import StochasticTrajectory, TimeSeries, Position
from ..integrate import Position


class BrownianMotion(StochasticTrajectory):
    """
    (1d) Brownian motion.

    The equations of motion are given by: $$m\ddot x = -\gamma \dot x
    - V'(x) + \sqrt{2\gamma \beta^{-1}} dW_t,$$ where

    :param x: is the position of the particle
    :param beta: The thermodynamic beta $\beta^{-1} = k_B T$
    :param gamma: is the damping constant
    :param is_overdamped: is a boolean tag equivalent to taking the
        limit $m \to 0$.

        - V is the potential (V' denotes its first spatial
          derivative).  By default this is 0, though this is
          overridden in child classes.

        - dW_t is a white noise Wiener process with standard deviation
          dt.

    At a quick glance, one sees that this is overparametrized.  We
    subsume $m$ into a redefinition of the other parameters.  We set
    $m:=1$.  We can't change this value but we can access it.

    In the over-damped limit, $m\ddot x \ll 1 $, so the E.O.M reduces
    to: $$\gamma \dot x = - V'(x) + \sqrt{2\gamma k_B T} dW_t,$$
    """

    def __init__(self,
                 beta: float = 1.,
                 gamma: float = 1.,
                 is_overdamped: bool = False,
                 **kwargs):
        self.beta = beta
        self.gamma = gamma
        self.is_overdamped = is_overdamped

        vectorized = not is_overdamped
        n_dofs = 1 if is_overdamped else 2
        super().__init__(vectorized=vectorized, n_dofs=n_dofs, **kwargs)

        # We run the conditional here so we don't have to go through it every time we integrate a step forward
        if self.is_overdamped:
            self.m = 0.
            self._take_step = lambda t, x: - self._grad_potential(x) / self.gamma
            self._get_random_step = lambda t, x:  np.sqrt(2. * self.gamma / self.beta) / self.gamma
        else:
            self.m = 1.
            self._take_step = lambda t, x: np.array([-self.gamma * x[0] - self._grad_potential(x[1]), x[0]])
            self._get_random_step = lambda t, x: np.array([np.sqrt(2. * self.gamma / self.beta), 0])


    def __repr__(self):
        return f"<BrownianMotion beta:{self.beta} gamma:{self.gamma} is_overdamped:{self.is_overdamped} timestep={self.timestep}>"

    # Helper methods for computing various macroscopic observables
    # By default we assume a free particle

    @staticmethod
    def _grad_potential(x: Union[NDArray[Any], float]) -> Union[NDArray[Any, float], float]:
        return 0.

    @staticmethod
    def _potential_energy(x: Union[NDArray[Any], float]) -> Union[NDArray[Any, float], float]:
        return 0.

    def _kinetic_energy(self, v: Union[NDArray[Any], float]) -> Union[NDArray[Any, float], float]:
        return (self.m * v ** 2) / 2.

    # Wrappers for computing macroscopic observables of either
    # positions or whole timeseries

    def grad_potential(self, state: Union[TimeSeries, Position]) -> Union[NDArray[Any, float], float]:
        if isinstance(state, TimeSeries):
            # i.e. it's a time_series
            return self._grad_potential(state[:, 0])

        return self._grad_potential(state[0])

    def potential_energy(self, state: Union[TimeSeries, Position]) -> Union[NDArray[Any, float], float]:
        if isinstance(state, TimeSeries):
            # i.e. it's a time_series
            return self._potential_energy(state[:, 0])

        return self._potential_energy(state[0])

    def kinetic_energy(self, state: Union[TimeSeries, Position]) -> Union[NDArray[Any, float], float]:
        if isinstance(state, TimeSeries):
            # i.e. it's a time_series

            # we make sure we're in the not overdamped case
            if state.shape[1] > 1:
                return self._kinetic_energy(state[:, 1])

            # otherwise there is no notion of instantaneous velocity and kinetic energy
            return 0.

        return self._kinetic_energy(state[1])

    def energy(self, state: Union[TimeSeries, Position]) -> Union[NDArray[Any, float], float]:
        return self.potential_energy(state) + self.kinetic_energy(state)


    def boltzmann_weights(self, state: Union[TimeSeries, Position]) -> Union[NDArray[Any, float], float]:
        energies = self.energy(state)
        return np.exp(-energies * (self.beta))

    def boltzmann_probs(self, state: Union[TimeSeries, Position], dX: float=1) -> Union[NDArray[Any, float], float]:
        # TODO: Do something with dX
        weights = self.boltzmann_weights(state)
        return weights / np.sum(weights)

    def take_step(self, t: float, x: Position) -> Position:
        return self._take_step(t, x)

    def get_random_step(self, t: float, x: Position) -> Position:
        return self._get_random_step(t, x)

    @staticmethod
    def count_crossovers(time_series: TimeSeries,
                         position: Position,
                         min_staying_timesteps: int=10) -> int:
        """
        Counts the number of times the time_series crosses the point at
        the specified position.  To make sure we've really crossed
        (and don't immediately jump back), we specify a
        `min_staying_timesteps`: a time_series only counts as having
        crossed a point if it is still on the other side after this
        number of steps.

        :param time_series: A phase-space time-series
        :param position: A phase-space position
        :param min_staying_timesteps: Minimum number of timesteps
            to qualify as a "crossover"
        """
        binary_series = np.where(time_series < position, 1, 0)[:: min_staying_timesteps]
        crossovers = np.where(binary_series != np.roll(binary_series, -1), 1, 0)
        return np.sum(crossovers)

    @staticmethod
    def running_avg(time_series: TimeSeries, fn: Callable[[TimeSeries], NDArray[Any, float]], window_size: int=1000, step_size: int=100, verbose: bool=False) -> NDArray[Any, float]:
        n_windows = int(np.floor((time_series.shape[0] - window_size + 1) / step_size))
        avgs = np.zeros(n_windows)

        range_ = range(n_windows)

        if verbose:
            range_ = tqdm(range_, desc="Calculating running average")

        for i in range_:
            avgs[i] = np.mean(fn(time_series[i * step_size: i * step_size+ window_size]))

        return avgs

class DoubleWell(BrownianMotion):
    """
    :param state: refers to a full state in phase-space.  This is a pair
              [x, v] for the not overdamped case.
    :param x: is the spatial component of the state.  This is the full
              state in the overdamped case.
    :param v: is the velocity component of the state.
    """

    # Helper methods for computing various macroscopic observables

    @staticmethod
    def _grad_potential(x: Union[NDArray[Any], float]) -> float:
        return 4. * x * (x ** 2 - 1)

    @staticmethod
    def _potential_energy(x: Union[NDArray[Any], float]) -> float:
        return (x ** 2 - 1.) ** 2

    @property
    def dominant_eigval(self):
        return ((np.sqrt(17) - 1) * np.exp(- self.beta) / (np.pi * np.sqrt(2)))

    def get_timescale(self, eigval: float, multiplier: float=1.) -> float:
        return -float(self.timestep * multiplier/ np.log(eigval))

    def transition_time(self, trajectory: TimeSeries) -> float:
        trajectory_duration = trajectory.shape[0] * self.timestep
        n_crossovers = self.count_crossovers(trajectory, 0.)
        return trajectory_duration / n_crossovers
