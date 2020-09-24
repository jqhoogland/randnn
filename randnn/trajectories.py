"""

Contains baseline wrappers for generating phase-space trajectories,
most importantly a wrapper for performing stochastic integration using the
Euler Maruyama scheme.

Author: Jesse Hoogland
Year: 2020

"""
import os, hashlib, logging
from typing import List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm
from scipy.integrate import RK45
from .integrate import EulerMaruyama
from .utils import np_cache, qr_positive, random_orthonormal


class Trajectory:
    """

    A wrapper for an ODESolver to integrate formulas specified in children.

    """
    def __init__(self,
                 integrate,
                 init_state: Optional[Union[np.ndarray, float]] = None,
                 n_dofs: Optional[int] = 100):
        """
        :param init_state: the state to initialize the neurons with.
            defaults to a state of `n_dofs` neurons drawn randomly from the uniform distribution.
            if int, then this multiplies the above,
            if left blank, then `n_dofs` must be specified.
        :param n_dofs: the number of dofs.
            if `init_state` is of type `int`, this must be specified,
            else `n_dofs` is overwritten by the size of `init_state`
        """

        self.integrate = integrate

        if init_state is None:
            assert not n_dofs is None
            init_state = np.random.uniform(size=n_dofs)
        elif isinstance(init_state, float):
            assert not n_dofs is None
            init_state = np.random.uniform(size=n_dofs) * init_state
        else:
            n_dofs = init_state.size

        self.init_state = init_state
        self.n_dofs = n_dofs

    def __str__(self):
        return "trajectory-dof{}".format(self.n_dofs)

    def take_step(self, t: int, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @np_cache(dir_path="./saves/lyapunov/", file_prefix="spectrum-")
    def get_lyapunov_spectrum(self,
                              trajectory: np.ndarray,
                              n_burn_in: int = 0,
                              n_exponents: Optional[int] = None,
                              t_ons: int = 10) -> np.ndarray:
        """
        :param trajectory: the discretized samples, with shape (n_timesteps, n_dofs),
        :param n_burn_in: the number of initial transients to discard
        :param n_exponents: the number of lyapunov exponents to calculate (in decreasing order).
            Leave this blank to compute the full spectrum.

        TODO: Iteratively compute the optimal `t_ons`
        """

        if n_exponents is None:
            n_exponents = self.n_dofs

        assert n_exponents <= self.n_dofs

        # Decompose the growth rates using the QR decomposition
        lyapunov_spectrum = np.zeros(n_exponents)

        # We renormalize (/sample) only once every `t_ons` steps
        n_samples = (trajectory.shape[0] - n_burn_in) // t_ons

        # q will update at each timestep
        # r will update only every `t_ons` steps
        q = random_orthonormal([self.n_dofs, n_exponents])
        r = np.zeros([n_exponents, n_exponents])

        # Burn in so Q can relax to the Osedelets matrix
        for t, state in tqdm(enumerate(trajectory[:n_burn_in]),
                             desc="Burning-in Osedelets matrix"):
            q = self.jacobian(state) @ q
            if (t % t_ons == 0):
                q, _ = qr_positive(q)

        # Run the actual decomposition on the remaining steps
        for t, state in tqdm(enumerate(trajectory[n_burn_in:]),
                             desc="QR-Decomposition of trajectory"):
            q = self.jacobian(state) @ q

            if (t % t_ons == 0):
                q, r = qr_positive(q)

                r_diagonal = np.copy(np.diag(r))
                r_diagonal[r_diagonal == 0] = 1

                lyapunov_spectrum += np.log(r_diagonal)

        # The Lyapunov exponents are the time-averaged logarithms of the
        # on-diagonal (i.e scaling) elements of R
        lyapunov_spectrum /= n_samples

        return lyapunov_spectrum

    @np_cache(dir_path="./saves/trajectories/", file_prefix="trajectory-")
    def run(self, n_burn_in: int = 500, n_steps: int = 10000):

        integrator = self.integrate(self.init_state, n_steps)
        state = np.zeros([n_steps, self.n_dofs])

        for _ in tqdm(range(n_burn_in), desc="Burning in"):
            integrator.step()

        for t in tqdm(range(n_steps), desc="Generating samples: "):
            state[t, :] = np.array(integrator.y)
            integrator.step()

        return state


class DeterministicTrajectory(Trajectory):
    def __init__(self, max_step=0.01, vectorized=True, **kwargs):
        integrate = lambda init_dofs, n_steps: RK45(self.take_step,
                                                    0,
                                                    init_dofs,
                                                    n_steps,
                                                    max_step=max_step,
                                                    vectorized=vectorized)
        super(DeterministicTrajectory, self).__init__(integrate=integrate,
                                                      **kwargs)


class StochasticTrajectory(Trajectory):
    def __init__(self, step_size=0.001, vectorized=True, **kwargs):
        self.step_size = step_size
        integrate = lambda init_dofs, n_steps: EulerMaruyama(
            self.take_step,
            self.get_random_step,
            0,
            init_dofs,
            n_steps,
            step_size=step_size,
            vectorized=vectorized)

        super(StochasticTrajectory, self).__init__(integrate=integrate,
                                                   **kwargs)

    def get_random_step(self, t: int, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def downsample(rate: int = 10):
    """
    A function decorator that downsamples the result returned by that function.
    Assumes the function returns a np.ndarray of the shape (n_samples, n_dofs).

    :returns: a function which returns a downsampled array of shape (n_samples // rate, n_dofs)
    """
    def inner(func):
        def wrapper(*args, **kwargs):
            samples = func(*args, **kwargs)
            return samples[::rate]

        return wrapper

    return inner


def downsample_split(rate: int = 10):
    """
    A function decorator that downsamples the result returned by that function.
    Assumes the function returns a np.ndarray of the shape (n_samples, n_dofs).

    Instead of throwing away the intermediate entries, this creates a separate
    downsampled chain for each possible starting point.

    :returns: a function which returns a downsampled array of shape (rate, n_samples // rate, n_dofs)
    """
    def inner(func):
        def wrapper(*args, **kwargs):
            samples = func(*args, **kwargs)
            n_downsamples = len(samples) // rate
            downsamples = np.zeros((rate, n_downsamples, samples.shape[1]))

            for i in range(rate):
                downsamples[i, :, :] = samples[i::rate, :]

            return downsamples

        return wrapper

    return inner


def avg_over(key: str, axis: int = 0, include_std: bool = False):
    """
    A function decorator which averages a function over one of its given parameters.

    To be used in conjunction with the above.

    :param axis: the axis of the result to average over, defaults to 0, the first axis.
    :param kwargs: should contain one keyword argument

    e.g. A function is given an array [n_samples, n_timesteps, n_dofs].
    This decorator performs that function n_samples times for the values [i, :, :] where i in n_samples.
    """
    def inner(func):
        def wrapper(*args, **kwargs):
            value = kwargs.pop(key)

            # Make sure kwargs contains a suitable value for this key
            if value is None:
                raise ValueError(f"key {key} not in kwargs.")
            elif not isinstance(value, np.ndarray):
                raise TypeError(f"key {key} not a numpy array.")

            if axis != 0:
                np.swapaxes(value, 0, axis)

            responses = []

            for i in range(value.shape[0]):
                avg_kwarg = {}
                avg_kwarg[key] = value[i, :, :]
                responses.append(func(*args, **avg_kwarg, **kwargs))

            responses = np.array(responses)

            if include_std:
                return np.mean(responses, axis=0), np.std(responses, axis=axis)

            return np.mean(responses, axis=0)

        return wrapper

    return inner
