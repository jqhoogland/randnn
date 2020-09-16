"""

Contains code to determine the Lyapunov spectrum via the reorthonormalization procedure

Author: Jesse Hoogland
Year: 2020

See [@wolf1985]


"""
import numpy as np
from typing import Tuple

from tqdm import tqdm


def random_orthonormal(shape: Tuple[int, int]):
    # Source: https://stackoverflow.com/a/38430739/1701415
    A = np.random.randn(*shape)
    Q, R = np.linalg.qr(A)
    return Q @ Q.T


def get_lyapunov_spectrum(jacobians: np.ndarray) -> np.ndarray:
    n_timesteps, n_dofs, _ = jacobians.shape

    # Evolve an initially orthonormal system by repeated application of the Jacobian
    evolution = np.zeros((n_timesteps + 1, n_dofs, n_dofs))
    evolution[0, :, :] = random_orthonormal((n_dofs, n_dofs))
    for t, jacobian in tqdm(enumerate(jacobians),
                            desc="Computing linear evolution"):
        evolution[t + 1, :, :] = jacobian @ evolution[t]

    # Decompose the growth rates using the QR decomposition
    qs = np.zeros(evolution.shape)
    rs = np.zeros(evolution.shape)
    for t, state in tqdm(enumerate(evolution), desc="Decomposing evolution"):
        #print(state)
        qs[t, :, :], rs[t, :, :] = np.linalg.qr(state)

    # The Lyapunov exponents are the time-averaged logarithms of the on-diagonal (i.e scaling)
    # elements of R
    lyapunov_exponents = np.mean(np.log(
        np.abs(np.diagonal(rs, axis1=1, axis2=2))),
                                 axis=0)

    # We order these exponents in decreasing order
    lyapunov_exponents_ordered = np.sort(lyapunov_exponents)[::-1]

    return lyapunov_exponents_ordered
