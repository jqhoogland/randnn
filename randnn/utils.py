"""

Contains helper functions for other modules.

Author: Jesse Hoogland
Year: 2020

"""

import os, hashlib, logging
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp


def np_cache(dir_path: str = "./saves/", file_prefix: Optional[str] = None):
    """
    A wrapper to load a previous response to a function (or to run the function and save the result otherwise).
    Assuming the function returns a np.ndarray as its response
    """
    def inner(func):
        def wrapper(*args, **kwargs):
            params = (str(args) + str(kwargs)).encode('utf-8')
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


def eigsort(A, k, which="LM"):
    eig_vals, eig_vecs = sp.linalg.eigs(A, k, which=which)
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
