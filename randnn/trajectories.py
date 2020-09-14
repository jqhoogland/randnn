"""

Contains baseline wrappers for generating phase-space trajectories,
most importantly a wrapper for performing stochastic integration using the
Euler Maruyama scheme.

Author: Jesse Hoogland
Year: 2020

"""
import os
from typing import List

import numpy as np
from tqdm import tqdm
from scipy.integrate import RK45
from .integrate import EulerMaruyama


class Trajectory:
    """

    A wrapper for an ODESolver to integrate formulas specified in children.

    """
    def __init__(self, n_dofs=3):
        self.n_dofs = n_dofs

    def __str__(self):
        return "trajectory-dof{}".format(self.n_dofs)

    def take_step(self, t: int, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def run(self, init_dofs=None, n_burn_in=500, n_steps=10000, max_step=0.02):
        if init_dofs == None:
            init_dofs = np.random.uniform(size=self.n_dofs)

        integrator = self.integrand(init_dofs, n_steps)
        state = np.zeros([n_steps, self.n_dofs])

        for _ in tqdm(range(n_burn_in), desc="Burning in"):
            integrator.step()

        for t in tqdm(range(n_steps), desc="Generating samples: "):
            state[t, :] = np.array(integrator.y)
            integrator.step()

        return state

    def run_or_load(self,
                    filename=None,
                    init_dofs=None,
                    n_burn_in=500,
                    n_steps=10000,
                    max_step=0.02):
        trajectory = self.load(filename)

        if trajectory.size == 0:
            trajectory = self.gen_data(init_dofs=init_dofs,
                                       n_burn_in=n_burn_in,
                                       n_steps=n_steps,
                                       max_step=max_step)

        return trajectory

    def save(self, trajectory, filename=None):
        if filename is None:
            filename = "./saves/{}.npy".format(self.__str__())

        np.save(filename, trajectory)

    def load(self, filename=None):
        if filename is None:
            filename = "./saves/{}.npy".format(self.__str__())

        if os.path.isfile(filename):
            return np.load(filename)
        else:
            return np.array([])


class DeterministicTrajectory(Trajectory):
    def __init__(self, max_step=0.01, vectorized=True, n_dofs=3):
        super(DeterministicTrajectory, self).__init__(n_dofs=n_dofs)
        self.integrate = lambda init_dofs, n_steps: RK45(self.take_step,
                                                         0,
                                                         init_dofs,
                                                         n_steps,
                                                         max_step=max_step,
                                                         vectorized=vectorized)


class StochasticTrajectory(Trajectory):
    def __init__(self, step_size=0.001, vectorized=True, n_dofs=3):
        super(StochasticTrajectory, self).__init__(n_dofs=n_dofs)
        self.step_size = step_size
        self.integrate = lambda init_dofs, n_steps: EulerMaruyama(
            self.take_step,
            self.get_random_step,
            0,
            init_dofs,
            n_steps,
            step_size=step_size,
            vectorized=vectorized)

    def get_random_step(self, t: int, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError
