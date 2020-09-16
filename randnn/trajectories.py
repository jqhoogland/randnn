"""

Contains baseline wrappers for generating phase-space trajectories,
most importantly a wrapper for performing stochastic integration using the
Euler Maruyama scheme.

Author: Jesse Hoogland
Year: 2020

"""
import os, hashlib, pickle
from typing import List, Optional

import numpy as np
from tqdm import tqdm
from scipy.integrate import RK45
from .integrate import EulerMaruyama


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
        return hashlib.md5(self.__repr__().encode('utf-8')).hexdigest()

    def take_step(self, t: int, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def run(self,
            n_burn_in: int = 500,
            n_steps: int = 10000,
            return_jacobians: bool = False):

        integrator = self.integrate(self.init_state, n_steps)
        state = np.zeros([n_steps, self.n_dofs])
        jacobians = np.zeros([n_steps, self.n_dofs, self.n_dofs])

        for _ in tqdm(range(n_burn_in), desc="Burning in"):
            integrator.step()

        for t in tqdm(range(n_steps), desc="Generating samples: "):
            state[t, :] = np.array(integrator.y)
            jacobians[t, :, :] = self.jacobian(integrator.y)
            integrator.step()

        if return_jacobians:
            return state, jacobians

        return state

    def run_or_load(self,
                    filename: Optional[str] = None,
                    n_burn_in: int = 500,
                    n_steps: int = 10000,
                    return_jacobians: bool = False):

        res = self.load(filename)

        if not res:
            res = self.run(n_burn_in=n_burn_in,
                           n_steps=n_steps,
                           return_jacobians=return_jacobians)

        return res

    def save(self,
             trajectory: np.ndarray,
             jacobians: Optional[np.ndarray] = None,
             filename: Optional[str] = None):

        if filename is None:
            filename = "./saves/{}.pickle".format(self.filename)

        with open(filename, "wb+") as handle:
            pickle.dump([trajectory, jacobians],
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filename: Optional[str] = None):
        if filename is None:
            filename = "./saves/{}.pickle".format(self.filename)

        if os.path.isfile(filename):
            with open(filename, 'rb') as handle:
                return pickle.load(handle)

        return []


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
