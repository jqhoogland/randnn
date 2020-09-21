"""

Contains baseline wrappers for generating phase-space trajectories,
most importantly a wrapper for performing stochastic integration using the
Euler Maruyama scheme.

Author: Jesse Hoogland
Year: 2020

"""
import os, hashlib
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm
from scipy.integrate import RK45
from .integrate import EulerMaruyama


def random_orthonormal(shape: Tuple[int, int]):
    # Source: https://stackoverflow.com/a/38430739/1701415
    a = np.random.randn(*shape)
    q, r = np.linalg.qr(a)
    return q @ q.T


def qr_positive(a: np.ndarray, *args,
                **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    q, r = np.linalg.qr(a, *args, **kwargs)
    diagonal_signs = np.sign(np.diagonal(r))
    return q * diagonal_signs, diagonal_signs * r  # TODO: make sure these are aligned correctly


class Trajectory:
    """

    A wrapper for an ODESolver to integrate formulas specified in children.

    """
    def __init__(self,
                 integrate,
                 init_state: Optional[np.ndarray] = None,
                 n_dofs: Optional[int] = 100):
        """
        :param init_state: the state to initialize the neurons with.
            defaults to a state of `n_dofs` neurons drawn randomly from the uniform distribution.
            if left blank, then `n_dofs` must be specified.
        :param n_dofs: the number of dofs.
            if `init_state` is of type `int`, this must be specified,
            else `n_dofs` is overwritten by the size of `init_state`
        """

        self.integrate = integrate

        if init_state is None:
            assert not n_dofs is None
            init_state = np.random.uniform(size=n_dofs)
        else:
            n_dofs = init_state.size

        self.init_state = init_state
        self.n_dofs = n_dofs

    def __str__(self):
        return "trajectory-dof{}".format(self.n_dofs)

    @property
    def filename(self):
        hashlib.md5(self.__repr__().encode('utf-8')).hexdigest()

    def take_step(self, t: int, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_lyapunov_spectrum(self,
                              trajectory: np.ndarray,
                              n_burn_in: int = 100) -> np.ndarray:
        # Decompose the growth rates using the QR decomposition
        t_ons = 10  # TODO: actually compute this

        n_samples = trajectory.shape[0] // t_ons
        lyapunov_spectrum = np.zeros(self.n_dofs)
        q = random_orthonormal([self.n_dofs, self.n_dofs])
        r = np.zeros([self.n_dofs, self.n_dofs])

        for t, state in tqdm(enumerate(trajectory[:n_burn_in]),
                             desc="Burning-in Osedelets matrix"):
            q = self.jacobian(state) @ q
            if (t % t_ons):
                q, _ = np.linalg.qr(q)

        for t, state in tqdm(enumerate(trajectory[n_burn_in:]),
                             desc="QR-Decomposition of trajectory"):
            q = self.jacobian(state) @ q
            if (t % t_ons):
                q, r = qr_positive(q, mode="complete")
                #print(q.shape, r.shape)
                lyapunov_spectrum += np.log(np.diagonal(r))

        # The Lyapunov exponents are the time-averaged logarithms of the on-diagonal (i.e scaling)
        # elements of R
        lyapunov_spectrum /= n_samples

        # We order these exponents in decreasing order
        lyapunov_spectrum_ordered = np.sort(lyapunov_spectrum)[::-1]

        return lyapunov_spectrum_ordered

    def run(self, n_burn_in: int = 500, n_steps: int = 10000):

        integrator = self.integrate(self.init_state, n_steps)
        state = np.zeros([n_steps, self.n_dofs])

        for _ in tqdm(range(n_burn_in), desc="Burning in"):
            integrator.step()

        for t in tqdm(range(n_steps), desc="Generating samples: "):
            state[t, :] = np.array(integrator.y)
            integrator.step()

        return state

    def run_or_load(self,
                    filename: Optional[str] = None,
                    n_burn_in: int = 500,
                    n_steps: int = 10000):

        res = self.load(filename)

        if res.size:
            res = self.run(
                n_burn_in=n_burn_in,
                n_steps=n_steps,
            )

        return res

    def save(self,
             trajectory: np.ndarray,
             filename: Optional[str] = None) -> None:

        if filename is None:
            filename = "./saves/{}.npy".format(self.filename)

        return np.save(filename, trajectory)

    def load(self, filename: Optional[str] = None) -> np.ndarray:
        if filename is None:
            filename = "./saves/{}.npy".format(self.filename)

        if os.path.isfile(filename):
            return np.load(filename)

        return np.array([])


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
