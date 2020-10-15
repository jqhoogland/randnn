"""

Contains basic functions for plotting various elements of this project:
- (low-dimensional projections of) trajectories and their averages
- Lyapunov spectra
- etc.

Author: Jesse Hoogland
Year: 2020

"""
from collections.abc import Sequence
from typing import Union
import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, rc
from tqdm import tqdm

from .transfer_operator import TransferOperator
from .networks import ContinuousNN
from .utils import *

plt.rc('text', usetex=True)


def plot_trajectory_avg(trajectory):
    avg_trajectory = np.mean(trajectory, axis=1)
    stdev_trajectory = np.std(trajectory, axis=1)
    plt.errorbar(np.arange(avg_trajectory.size),
                 avg_trajectory,
                 yerr=stdev_trajectory)


def plot_trajectory_samples(trajectory: np.ndarray,
                            indices: Union[int, np.ndarray] = 5):
    """
    :param trajectory: of shape (n_timesteps, n_dofs)
    :param indices: if int, plots the trajectories of the first `indices` many neurons on top of eachother.
        if np.ndarray, plots the trajectories of the neurons corresponding to the indices in the array.
    """

    try:
        # indices is np.ndarray
        for i in range(indices.size):
            plt.plot(trajectory[:, indices[i]], color="tab:blue")
    except AttributeError:
        # indices is int
        for i in range(indices):
            plt.plot(trajectory[:, i], color="tab:blue")

    plt.title("Sample neural trajectories")
    plt.xlabel("Time ($\tau$)")
    plt.ylabel("Activity")


def plot_random_matrix_spectrum(matrix, radius=None):
    N, _ = matrix.shape    # Assuming square
    if radius is None:
        radius = np.std(matrix) * np.sqrt(N)

    C = np.zeros((N, N), dtype=np.complex_)
    eigs, _ = np.linalg.eig(matrix + C)

    #print(eigs)
    fix, ax = plt.subplots()
    ax.scatter(eigs.real, eigs.imag, color=(0, 0, 0), s=2, alpha=0.5)
    circle = patches.Circle((0, 0),
                            radius,
                            fill=False,
                            color="k",
                            linewidth=2,
                            edgecolor='b')

    plt.axis("square")
    ax.add_patch(circle)
    ax.axvline(x=1., linestyle="--")

    ax.set_title("Eigenvalue spectrum of the connectivity matrix $J_{ij}$")
    ax.set_ylabel("Complex part")
    ax.set_xlabel("Real part")


def plot_lyapunov_spectrum(lyapunov_spectrum, title="Lyapunov spectrum"):
    plt.plot(lyapunov_spectrum)
    plt.axhline(y=0., linestyle="--")
    plt.title(title)
    plt.ylabel("Lyapunov exponent")
    plt.xlabel("Index (decreasing order)")


def plot_eig_spectrum(spectrum,
                      eigs_range,
                      title="Eigenvalue spectrum",
                      label=""):
    plt.plot(1 + np.arange(*eigs_range),
             spectrum[eigs_range[0]:eigs_range[1]],
             label=label)
    plt.title(title)
    plt.xlabel("Eigenvalue index")
    plt.ylabel("Absolute value.")


def plot_t_imp_scaling(
        time_series: np.ndarray,
        eigval_idx: int,
        n_clusters_list: Sequence = [2, 10,
                                     100],    # TODO: 3.9 - Sequence[int]
        transition_times: Sequence = range(1, 30, 2),
        timestep: float = 1,
        labeling_method: str = "kmeans"):
    """
    Plot the scaling of the implied timescale for different numbers of clusters.

    The eigenvalues of the transfer matrix correspond to unique (implied) timescales.
    They are related by $t_\text{imp} = -\tau / \log|\lambda|$, where:
    - $t_\text{imp}$ is the implied timescale,
    - $\tau$ is the discretization timestep, and
    - $\lambda$ is the eigenvalue for which we want the corresponding timescale.

    :param time_series: the evolution under consideration
    :param eigval_idx: the index of the eigenvalue (in decreasing order) whose implied timescale interests us
    :param n_clusters_list: a list of numbers of clusters. We plot a single line for each of these.
    :param transition_timescales: the number of frames to include in computing the transfer matrix.
        These are the points that will constitute the $x$-axis.
    :param timestep: this is the discretization time, a relic of numerically approximating a continuous system.
    """
    transfer_operator = TransferOperator(labeling_method=labeling_method)

    for n_clusters in n_clusters_list:
        # We plot a single curve for every choice of number of clusters
        t_imps = np.zeros(len(transition_times))
        for i, transition_time in tqdm(
                enumerate(transition_times),
                desc="computing `t_imp` for `n_clusters = {}`".format(
                    n_clusters)):
            # Each curve has its timescale sampled at the points defined in transition_timescales
            t_imps[i] = transfer_operator.get_t_imp(
                time_series,
                [
                    eigval_idx    # get_t_imp can sample multiple eigenvalues at the same time
                ],
                n_clusters,
                transition_time,
                timestep
            )[0]    # no surprise, the method returns an array of eigenvalues.

        # We label each curve by the number of clusters, $n_p$, it corresponds to.
        plt.plot(transition_times,
                 t_imps,
                 label="$n_p = {}$".format(n_clusters))

    plt.title("Scaling of $t$ with $\\tau$ and $n_p$")
    plt.legend()
    plt.show()


def plot_max_l_with_g(gs: Sequence,
                      n_dofs: int = 100,
                      timestep: float = 0.1,
                      n_steps: int = 10000,
                      n_burn_in: int = 1000,
                      t_ons: int = 10):
    max_lyapunov_exps = np.zeros(len(gs))

    for i, g in enumerate(gs):
        logging.info(
            "Deriving maximum lyapunov exponent for `g = {}`".format(g))
        cont_nn = ContinuousNN(coupling_strength=g,
                               n_dofs=n_dofs,
                               max_step=timestep)
        trajectory = cont_nn.run(n_steps=n_steps, n_burn_in=n_burn_in)
        lyapunov_spectrum = cont_nn.get_lyapunov_spectrum(trajectory, t_ons=10)
        max_lyapunov_exps[i] = lyapunov_spectrum[0]

    plt.plot(gs, max_lyapunov_exps)
    plt.plot(gs, np.zeros(len(gs)), ":")


def plot_trivial_fixed_pt_with_g(gs: Sequence,
                                 n_dofs: int = 100,
                                 timestep: float = 0.1,
                                 n_steps: int = 10000,
                                 n_burn_in: int = 1000,
                                 t_ons: int = 10,
                                 atol: float = 1e-3):
    trivial_fixed_pt_proportions = np.zeros(len(gs))

    for i, g in enumerate(gs):
        logging.info("Deriving fraction at 0 for `g = {}`".format(g))
        cont_nn = ContinuousNN(coupling_strength=g,
                               n_dofs=n_dofs,
                               max_step=timestep)
        trajectory = cont_nn.run(n_steps=n_steps, n_burn_in=n_burn_in)

        trivial_fixed_pt_proportions[i] = count_trivial_fixed_pts(
            trajectory.T, atol) / n_dofs

    plt.title(
        "The proportion of neurons at the trivial fixed point with coupling strength"
    )
    plt.xlabel("$g$, the coupling strength")
    plt.ylabel("Fraction of neurons at the trivial fixed point")
    plt.plot(gs, trivial_fixed_pt_proportions)


def plot_nontrivial_fixed_pt_with_g(gs: Sequence,
                                    n_dofs: int = 100,
                                    timestep: float = 0.1,
                                    n_steps: int = 10000,
                                    n_burn_in: int = 1000,
                                    t_ons: int = 10,
                                    atol: float = 1e-3):
    fixed_pt_proportions = np.zeros(len(gs))

    for i, g in enumerate(gs):
        logging.info("Deriving fraction at 0 for `g = {}`".format(g))
        cont_nn = ContinuousNN(coupling_strength=g,
                               n_dofs=n_dofs,
                               max_step=timestep)
        trajectory = cont_nn.run(n_steps=n_steps, n_burn_in=n_burn_in)

        n_trivial_fixed_pts = count_trivial_fixed_pts(trajectory.T, atol)
        n_fixed_pts = count_fixed_pts(trajectory.T, atol)

        fixed_pt_proportions[i] = (n_fixed_pts - n_trivial_fixed_pts) / n_dofs

    plt.title(
        "The proportion of neurons at non-trivial fixed points with coupling strength"
    )
    plt.xlabel("$g$, the coupling strength")
    plt.ylabel("Fraction of neurons at non-trivial fixed points")
    plt.plot(gs, fixed_pt_proportions)


def plot_cycles_with_g(gs: Sequence,
                       n_dofs: int = 100,
                       timestep: float = 0.1,
                       n_steps: int = 10000,
                       n_burn_in: int = 1000,
                       t_ons: int = 10,
                       atol: float = 1e-3,
                       max_n_steps: int=10000):
    cycle_proportions = np.zeros(len(gs))

    for i, g in enumerate(gs):
        logging.info("Deriving fraction at 0 for `g = {}`".format(g))
        cont_nn = ContinuousNN(coupling_strength=g,
                               n_dofs=n_dofs,
                               max_step=timestep)
        trajectory = cont_nn.run(n_steps=n_steps, n_burn_in=n_burn_in)

        n_cycles = count_cycles(trajectory.T, atol, max_n_steps)
        cycle_proportions[i] = n_cycles / n_dofs

    plt.title(
        "The proportion of neurons in a regular cycle with coupling strength")
    plt.xlabel("$g$, the coupling strength")
    plt.ylabel("Fraction of neurons in cycles")
    plt.plot(gs, cycle_proportions)
