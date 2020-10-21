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
                            indices: Union[int, np.ndarray] = 5,
                            title: str = "Sample trajectories"):
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

    plt.title(title)
    plt.xlabel("Time ($\\tau$)")
    plt.ylabel("Activity")


def plot_samples(
    coupling_strength: float,
    number: int = 5,
    n_dofs: int = 100,
    timestep: float = 0.1,
    n_steps: int = 10000,
    n_burn_in: int = 1000,
    t_ons: int = 10,
):
    cont_nn = ContinuousNN(coupling_strength=coupling_strength,
                           n_dofs=n_dofs,
                           timestep=timestep)
    trajectory = cont_nn.run(n_steps=n_steps, n_burn_in=n_burn_in)

    plot_trajectory_samples(
        trajectory,
        number,
        title="Sample trajectories at $g={}$".format(coupling_strength))


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
                      t_ons: int = 10,
                      network_seed: int=123):
    """
    Plot the maximum lyapunov exponent as a function of the coupling
    strength, $g$.

    :param gs: The list of coupling strengths for which to sample
        corresponding lyapunov exponents.
    :param n_dofs: The number of elements (neurons).
    :param timestep: The discretization timestep.
    :param n_steps: How many timesteps to simulate for each
        coupling_strength.
    :param n_burn_in: How many burnin timesteps to simulate before
        starting to record.
    :param t_ons: Akin to a downsampling time when computing the full
        Lyapunov spectrum.
    :param network_seed: The random seed to use when drawing coupling
        matrices.  Setting this ensures that the networks at different
        sizes differ only in the coupling strength and not in the
        normalized network topologies.  This lets us compare similar
        networks.
    """

    max_lyapunov_exps = np.zeros(len(gs))

    for i, g in enumerate(gs):
        logging.info(
            "Deriving maximum lyapunov exponent for `g = {}`".format(g))

        # 1. Initialize a network
        cont_nn = ContinuousNN(coupling_strength=g, n_dofs=n_dofs,
                               timestep=timestep,
                               network_seed=network_seed)

        # 2. Simulate a phase space trajectory
        trajectory = cont_nn.run(n_steps=n_steps, n_burn_in=n_burn_in)

        # 3. Derive the Lyapunov spectrum (using reorthonormalization)
        lyapunov_spectrum = cont_nn.get_lyapunov_spectrum(trajectory,
                                                          t_ons=10)

        # 4. Log the maximum Lyapunov exponent.
        max_lyapunov_exps[i] = lyapunov_spectrum[0]

    plt.plot(gs, max_lyapunov_exps)
    plt.plot(gs, np.zeros(len(gs)), ":")


def plot_trivial_fixed_pt_with_g(gs: Sequence, n_dofs: int = 100,
                                 timestep: float = 0.1, n_steps: int =
                                 10000, n_burn_in: int = 1000, atol:
                                 float = 1e-3, network_seed: int=123):
    """
    Plot the fraction of dofs which settle to the trivial fixed point
    (=0) as a function of the coupling strength, $g$.

    :param gs: The list of coupling strengths for which to sample
        corresponding lyapunov exponents.
    :param n_dofs: The number of elements (neurons).
    :param timestep: The discretization timestep.
    :param n_steps: How many timesteps to simulate for each
        coupling_strength.
    :param n_burn_in: How many burnin timesteps to simulate before
        starting to record.
    :param atol: This is the absolute numerical tolerance that
        determines whether a subtrajectory has reached 0.
    :param network_seed: The random seed to use when drawing coupling
        matrices.  Setting this ensures that the networks at different
        sizes differ only in the coupling strength and not in the
        normalized network topologies.  This lets us compare similar
        networks.
    """

    trivial_fixed_pt_proportions = np.zeros(len(gs))

    for i, g in enumerate(gs):
        logging.info("Deriving fraction at 0 for `g = {}`".format(g))

        # 1. Initialize a network
        cont_nn = ContinuousNN(coupling_strength=g, n_dofs=n_dofs,
                               timestep=timestep,
                               network_seed=network_seed)

        # 2. Simulate a phase space trajectory
        trajectory = cont_nn.run(n_steps=n_steps, n_burn_in=n_burn_in)

        # 3. Compute the fraction of dofs at 0
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
                                    atol: float = 1e-3,
                                    network_seed: int=123):
    """
    Plot the fraction of dofs which settle to a nontrivial fixed point
    (i.e. other than 0) as a function of the coupling strength, $g$.

    :param gs: The list of coupling strengths for which to sample
        corresponding lyapunov exponents.
    :param n_dofs: The number of elements (neurons).
    :param timestep: The discretization timestep.
    :param n_steps: How many timesteps to simulate for each
        coupling_strength.
    :param n_burn_in: How many burnin timesteps to simulate before
        starting to record.
    :param atol: This is the absolute numerical tolerance that
        determines whether a subtrajectory has settled to a fixed
        point.
    :param network_seed: The random seed to use when drawing coupling
        matrices.  Setting this ensures that the networks at different
        sizes differ only in the coupling strength and not in the
        normalized network topologies.  This lets us compare similar
        networks.
    """

    fixed_pt_proportions = np.zeros(len(gs))

    for i, g in enumerate(gs):
        logging.info(
            "Deriving fraction at non-0 fixed points for `g = {}`".format(g))

        # 1. Initialize a network
        cont_nn = ContinuousNN(coupling_strength=g, n_dofs=n_dofs,
                               timestep=timestep,
                               network_seed=network_seed)

        # 2. Simulate a phase space trajectory
        trajectory = cont_nn.run(n_steps=n_steps, n_burn_in=n_burn_in)

        # 3. Compute the (1) total number of of fixed points and (2) the number of trivial fixed points
        n_trivial_fixed_pts = count_trivial_fixed_pts(trajectory.T, atol)
        n_fixed_pts = count_fixed_pts(trajectory.T, atol)

        # 4. Compute the number of nontrivial fixed points from their
        # difference
        fixed_pt_proportions[i] = (n_fixed_pts - n_trivial_fixed_pts) / n_dofs

    plt.title(
        "The proportion of neurons at non-trivial fixed points with coupling strength"
    )
    plt.xlabel("$g$, the coupling strength")
    plt.ylabel("Fraction of neurons at non-trivial fixed points")
    plt.plot(gs, fixed_pt_proportions)


def plot_cycles_with_g(gs: Sequence, n_dofs: int = 100, timestep:
                       float = 0.1, n_steps: int = 10000, n_burn_in:
                       int = 1000, atol: float = 1e-3, max_n_steps:
                       int = 10000, network_seed: int=123):
    """
    Plot the fraction of dofs which settle to an oscillatory cycle
    (i.e. neither noisy behavior nor fixed points) as a function of
    the coupling strength, $g$.

    :param gs: The list of coupling strengths for which to sample
        corresponding lyapunov exponents.
    :param n_dofs: The number of elements (neurons).
    :param timestep: The discretization timestep.
    :param n_steps: How many timesteps to simulate for each
        coupling_strength.
    :param n_burn_in: How many burnin timesteps to simulate before
        starting to record.
    :param atol: This is the absolute numerical tolerance that
        determines whether a subtrajectory has settled into a cycle.
    :param network_seed: The random seed to use when drawing coupling
        matrices.  Setting this ensures that the networks at different
        sizes differ only in the coupling strength and not in the
        normalized network topologies.  This lets us compare similar
        networks.
    """


    cycle_proportions = np.zeros(len(gs))

    for i, g in enumerate(gs):
        logging.info("Counting cycles for `g = {}`".format(g))

        # 1. Initialize a network
        cont_nn = ContinuousNN(coupling_strength=g, n_dofs=n_dofs,
                               timestep=timestep,
                               network_seed=network_seed)

        # 2. Simulate a phase space trajectory
        trajectory = cont_nn.run(n_steps=n_steps, n_burn_in=n_burn_in)

        # 3. Compute the number of cycles and register the proportion
        # of dofs in a cycle.
        n_cycles = count_cycles(trajectory.T, atol, max_n_steps)
        cycle_proportions[i] = n_cycles / n_dofs

    plt.title(
        "The proportion of neurons in a regular cycle with coupling strength")
    plt.xlabel("$g$, the coupling strength")
    plt.ylabel("Fraction of neurons in cycles")
    plt.plot(gs, cycle_proportions)


def plot_participation_ratio_with_g(gs: Sequence,
                                    n_dofs: int = 100,
                                    timestep: float = 0.1,
                                    n_steps: int = 10000,
                                    n_burn_in: int = 1000,
                                    max_n_steps: int = 10000,
                                    network_seed: int=123):
    """
    Plot the maximum lyapunov exponent as a function of the coupling
    strength, $g$.

    :param gs: The list of coupling strengths for which to sample
        corresponding lyapunov exponents.
    :param n_dofs: The number of elements (neurons).
    :param timestep: The discretization timestep.
    :param n_steps: How many timesteps to simulate for each
        coupling_strength.
    :param n_burn_in: How many burnin timesteps to simulate before
        starting to record.
    :param max_n_steps: The number of samples (drawn from the end of
        the trajectory) over which to compute a PCA for the
        participation ratio.  Very long trajectory lengths may be more
        than the computer can reasonably handle.
    :param network_seed: The random seed to use when drawing coupling
        matrices.  Setting this ensures that the networks at different
        sizes differ only in the coupling strength and not in the
        normalized network topologies.  This lets us compare similar
        networks.
    """


    ratios = np.zeros(len(gs))

    for i, g in enumerate(gs):
        logging.info("Deriving participation ratio for `g = {}`".format(g))

        # 1. Initialize a network
        cont_nn = ContinuousNN(coupling_strength=g,
                               n_dofs=n_dofs,
                               timestep=timestep,
                               network_seed=network_seed)

        # 2. Simulate a phase space trajectory
        trajectory = cont_nn.run(n_steps=n_steps, n_burn_in=n_burn_in)

        # 3. Compute the participation ratio
        ratios[i] = participation_ratio(trajectory.T,
                                        max_n_steps=max_n_steps) / n_dofs

    plt.title("Relative Participation ratio, $D/N$")
    plt.xlabel("$g$, the coupling strength")
    plt.ylabel("Participation ratio, $D_{PCA}$")
    plt.plot(gs, ratios)
