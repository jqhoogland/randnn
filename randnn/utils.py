"""

Contains helper functions for other modules.

Author: Jesse Hoogland
Year: 2020

"""

import os, hashlib, logging
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

def np_cache(dir_path: str = "./saves/", file_prefix: Optional[str] = None, ignore: Optional[list]=[]):
    """
    A wrapper to load a previous response to a function (or to run the function and save the result otherwise).
    Assuming the function returns a np.ndarray as its response
    """
    def inner(func):
        def wrapper(*args, **kwargs):
            relevant_args = [ *args ]
            relevant_kwargs = { **kwargs}

            for ignore_arg in ignore:
                if isinstance(ignore_arg, int):
                    relevant_args.pop(ignore_arg)
                else:
                    del relevant_kwargs[ignore_arg]

            relevant_args = tuple(relevant_args)
            params = (str(relevant_args) + str(relevant_kwargs)).encode('utf-8')

            file_name = file_prefix + hashlib.md5(params).hexdigest() + ".npy"
            file_path = os.path.join(dir_path, file_name)

            if not os.path.isdir(dir_path):
                logging.info("Creating directory %s", dir_path)
                os.mkdir(dir_path)

            if os.path.isfile(file_path):
                logging.info("Loading from save %s", file_path)

                return np.load(file_path)

            response = func(*args, **kwargs)

            logging.info("Saving to %s", file_path)
            np.save(file_path, response)
            return response

        return wrapper

    return inner


def qr_positive(a: np.ndarray, *args,
                **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    q, r = np.linalg.qr(a, *args, **kwargs)
    diagonal_signs = np.sign(np.diagonal(r))
    return q @ np.diag(diagonal_signs), np.diag(
        diagonal_signs) @ r  # TODO: make sure these are aligned correctly


def random_orthonormal(shape: Tuple[int, int]):
    # Source: https://stackoverflow.com/a/38430739/1701415
    a = np.random.randn(*shape)
    q, r = qr_positive(a)
    return q


def eigsort(A, k, which="LM", eig_method="sp"):
    eig_vals, eig_vecs = None, None

    if (eig_method == "sp"):
        eig_vals, eig_vecs = sp.linalg.eigs(A, k, which=which)
    elif (eig_method == "np"):
        eig_vals, eig_vecs = np.linalg.eig(A)

    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]
    return (eig_vals, eig_vecs)


def normalize_rows(w):
    # Source: https://stackoverflow.com/a/59365444/1701415
    # Find the row scalars as a Matrix_(n,1)
    row_sum_w = sp.csr_matrix(w.sum(axis=1))
    row_sum_w.data = 1 / row_sum_w.data
    # Find the diagonal matrix to scale the rows
    row_sum_w = row_sum_w.transpose()
    scaling_matrix = np.diag(row_sum_w.toarray().reshape(
        (row_sum_w.shape[0] * row_sum_w.shape[1])))

    return scaling_matrix.dot(w)


def svd_whiten(X):
    # Source: https://stackoverflow.com/a/11336203/1701415

    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    # U and Vt are the singular matrices, and s contains the singular values.
    # Since the rows of both U and Vt are orthonormal vectors, then U * Vt
    # will be white
    X_white = np.dot(U, Vt)

    return X_white



def count_trivial_fixed_pts(trajectory: np.ndarray, atol: float = 1e-3) -> int:
    """
    :param trajectory: the trajectory of shape [n_dofs, n_timesteps]
    """

    # 1. We transform the trajectory into a binary array according to
    #    whether a point is within (=> 0)the given threshold of zeros or not (=> 1)
    trajectory_bin = np.where(np.isclose(trajectory, 0., atol=atol), 0, 1)

    # 2. We count the number of columns with only zeros.
    trajectory_collapsed = np.sum(trajectory_bin, axis=1)
    fixed_pt_neurons = np.where(trajectory_collapsed == 0, 1, 0)

    return np.sum(fixed_pt_neurons)


def count_fixed_pts(trajectory: np.ndarray, atol: float = 1e-3) -> int:
    """
    :param trajectory: the trajectory of shape [n_dofs, n_timesteps]
    """
    # 1. We transform the trajectory into a binary array according to
    #    whether a point is within (=> 0)the given threshold of its initial value or not (=> 1)

    initial_state = np.array([trajectory[:, 0]]).T * np.ones(trajectory.shape)
    trajectory_bin = np.where(np.isclose(trajectory, initial_state, atol=atol),
                              0, 1)

    # 2. We count the number of columns with only zeros.
    trajectory_collapsed = np.sum(trajectory_bin, axis=1)
    fixed_pt_neurons = np.where(trajectory_collapsed == 0, 1, 0)

    return np.sum(fixed_pt_neurons)


def count_cycles(trajectory: np.ndarray, atol: float = 1e-1, max_n_steps: Optional[int]=None) -> int:
    """
    :param trajectory: the trajectory of shape [n_dofs, n_timesteps]

    inspiration from https://stackoverflow.com/a/17090200/1701415
    """
    n_timesteps = trajectory.shape[1]

    if max_n_steps and max_n_steps < n_timesteps:
        trajectory = trajectory[:, (n_timesteps - max_n_steps):]

    cycles = np.zeros(trajectory.shape[0])

    # TODO: See if you can do this without the explicit for loop
    for i in tqdm(range(trajectory.shape[0]), desc="Counting cycles..."):
        path = trajectory[i, :]

        # 1. We compute the trajectories' autocorrelations (individually per neuron)
        path_normalized = path - np.mean(path)
        path_norm = np.sum(path_normalized ** 2)
        acor = np.correlate(path_normalized, path_normalized, "full") / path_norm
        acor = acor[len(acor) // 2:] # Autocorrelation is symmetrical about the half-way point
        # TODO: Use a more efficient way
        # TODO: figure out the effects of the boundary

        # 2. Figure out where the autocorrelation peaks are
        acor_peaks = np.where(np.logical_and(acor > np.roll(acor, 1),
                                acor > np.roll(acor, -1)), acor, 0)

        # 3. Figure out whether these peaks are within our tolerance of 1.
        #    We subtract one because the first entry will always have perfect autocorrelation.
        close_peaks = np.where(acor_peaks > 1. -atol, 1., 0.)
        is_cycle = np.sum(close_peaks) - 1. > 0

        cycles[i] = is_cycle

        # While debugging
        # plt.plot(acor)
        # plt.plot(close_peaks)
        # plt.show()

    return np.sum(cycles)


def test_count_trivial_fixed_pts():
    trajectory = np.array([[0., 1., 2.], [0., 0., 0.], [0., 0., 0.],
                           [1., 0., 0.]])
    assert count_trivial_fixed_pts(trajectory) == 2

def test_count_fixed_pts():
    trajectory = np.array([[0., 1., 2.], [0., 0., 0.], [2., 2., 2.],
                           [1., 1., 1.]])
    assert count_fixed_pts(trajectory) == 3

def test_count_cycles():
    x = np.arange(0, 100 * np.pi, 0.01)
    signal_1 = np.sin(x)
    signal_2 = np.random.uniform(size=len(x))
    signal_3 = np.sin(2 * x) + np.cos(3 * x)
    signal_4 = np.ones(len(x))
    signal_5 = np.random.uniform(size=len(x))

    trajectory = np.array([signal_1, signal_2, signal_3, signal_4, signal_5])

    assert count_cycles(trajectory, 0.1)  == 2


def test_qr_positive():
    a = np.random.uniform(size=(100, 50))
    q, r = qr_positive(a)

    logging.debug(a.shape, q.shape, r.shape)
    logging.debug(a, q @ r)

    assert np.allclose(a, q @ r)
    assert q.shape == (100, 50)
    assert np.allclose(q.T @ q, np.eye(50))
    assert r.shape == (50, 50)
    assert np.allclose(a, q @ r)
    assert np.all(np.diagonal(r) >= 0)


def test_random_orthonormal():
    q = random_orthonormal((100, 50))

    assert q.shape == (100, 50)
    assert np.allclose(q.T @ q, np.eye(50))


def test_normalize_rows():
    assert np.allclose(
        normalize_rows(
            sp.csr_matrix(
                np.array([[5, 1, 4], [0, 1, 1], [1, 1, 2]],
                         dtype="float64"))).todense(),
        sp.csr_matrix(
            np.array([[0.5, 0.1, 0.4], [0, 0.5, 0.5], [0.25, 0.25, 0.5]],
                     dtype="float64")).todense(),
    )
