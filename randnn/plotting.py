"""

Contains basic functions for plotting various elements of this project:
- (low-dimensional projections of) trajectories and their averages
- Lyapunov spectra
- etc.

Author: Jesse Hoogland
Year: 2020

"""
from collections.abc import Sequence
from typing import Union, Callable
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


def plot_samples(coupling_strength: float,
                 n_samples: int = 5,
                 n_dofs: int = 100,
                 timestep: float = 0.01,
                 n_steps: int = 10000,
                 n_burn_in: int = 1000,
                 n_sample_steps: int=10000,
                 network_seed: int = 123):
    """
    A function which plots trajectory samples.

    :param coupling_strength: The coupling strength for which to plot trajectories.
    :param n_samples: The number of dofs for which to display trajectories.
    :param n_dofs: The number of elements (neurons) to simulate.
    :param timestep: The discretization timestep.
    :param n_steps: How many timesteps to simulate for each
        coupling_strength.
    :param n_burn_in: How many burnin timesteps to simulate before
        starting to record.
    :param n_sample_steps: How many timesteps to use when drawing the trajectory
        (we take the last, not the first, n_sample_steps).
    :param network_seed: The random seed to use when drawing coupling
        matrices.  Setting this ensures that the networks at different
        sizes differ only in the coupling strength and not in the
        normalized network topologies.  This lets us compare similar
        networks.
    """
    assert n_sample_steps <= n_steps

    cont_nn = ContinuousNN(coupling_strength=coupling_strength,
                           n_dofs=n_dofs,
                           timestep=timestep, network_seed=network_seed)

    trajectory = cont_nn.run(n_steps=n_steps, n_burn_in=n_burn_in)[n_steps - n_sample_steps:]

    plot_trajectory_samples(
        trajectory,
        n_samples,
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


def plot_with_g(gs: Sequence,
                measure: Callable[[np.ndarray, ContinuousNN], float],
                n_dofs: int = 100,
                timestep: float = 0.1,
                n_steps: int = 10000,
                n_burn_in: int = 1000,
                t_ons: int = 10,
                normalize: bool = False,
                network_seed: int = 123):
    """
    A helper function for plotting measurements across ranges of
    coupling strength values.

    :param gs: The list of coupling strengths at which to measure sample trajectories.
    :param measure: A function which takes a generated trajectory and
        the Trajectory object that generated it, and computes a
        measurement on it.
    :param n_dofs: The number of elements (neurons).
    :param timestep: The discretization timestep.
    :param n_steps: How many timesteps to simulate for each
        coupling_strength.
    :param n_burn_in: How many burnin timesteps to simulate before
        starting to record.
    :param normalize: Whether or not to divide the measurement result
        by n_dofs.
    :param network_seed: The random seed to use when drawing coupling
        matrices.  Setting this ensures that the networks at different
        sizes differ only in the coupling strength and not in the
        normalized network topologies.  This lets us compare similar
        networks.
    """

    measurements = np.zeros(len(gs))

    for i, g in enumerate(gs):
        # 1. Initialize a network
        cont_nn = ContinuousNN(coupling_strength=g,
                               n_dofs=n_dofs,
                               timestep=timestep,
                               network_seed=network_seed)

        # 2. Simulate a phase space trajectory
        trajectory = cont_nn.run(n_steps=n_steps, n_burn_in=n_burn_in)

        # 3. Perform your measurement
        measurements[i] = measure(trajectory, cont_nn)

        # (4.) Potentially normalize
        if normalize:
            measurements[i] = measurements[i] / n_dofs

    plt.plot(gs, measurements)


def plot_max_l_with_g(gs: Sequence, t_ons: int = 10, **kwargs):
    """
    Plot the maximum lyapunov exponent as a function of the coupling
    strength, $g$.

    :param gs: The list of coupling strengths for which to sample
        corresponding lyapunov exponents.
    :param t_ons: Akin to a downsampling time when computing the full
        Lyapunov spectrum.
    :param kwargs: See `plot_with_g`.
    """

    # Derive the Lyapunov spectrum (using reorthonormalization) and
    # return its maximum exponent
    measure = lambda trajectory, cont_nn: cont_nn.get_lyapunov_spectrum(
        trajectory, t_ons=10)[0]

    plt.title("Maximum Lyapunov exponent as a function of coupling strength")
    plt.xlabel("$g$, the coupling strength")
    plt.ylabel("Maximum Lyapunov exponent, $\\lambda$")

    plot_with_g(gs, measure, **kwargs)
    plt.plot(gs, np.zeros(len(gs)), ":")


def plot_trivial_fixed_pt_with_g(gs: Sequence, atol: float = 1e-3, **kwargs):
    """
    Plot the fraction of dofs which settle to the trivial fixed point
    (=0) as a function of the coupling strength, $g$.

    :param gs: The list of coupling strengths for which to sample
        corresponding lyapunov exponents.
    :param atol: This is the absolute numerical tolerance that
        determines whether a subtrajectory has reached 0.
    :param kwargs: See `plot_with_g`.
    """

    #  Compute the fraction of dofs at 0
    measure = lambda trajectory, _: count_trivial_fixed_pts(trajectory, atol)

    plt.title(
        "The proportion of neurons at the trivial fixed point with coupling strength"
    )
    plt.xlabel("$g$, the coupling strength")
    plt.ylabel("Fraction of neurons at the trivial fixed point")
    plot_with_g(gs, measure, normalize=True, **kwargs)


def plot_nontrivial_fixed_pt_with_g(gs: Sequence,
                                    atol: float = 1e-3,
                                    **kwargs):
    """
    Plot the fraction of dofs which settle to a nontrivial fixed point
    (i.e. other than 0) as a function of the coupling strength, $g$.

    :param gs: The list of coupling strengths for which to sample
        corresponding lyapunov exponents.
    :param atol: This is the absolute numerical tolerance that
        determines whether a subtrajectory has settled to a fixed
        point.
    :param kwargs: See `plot_with_g`.
    """
    def measure(trajectory, *args):
        # Compute the (1) total number of of fixed points and (2) the
        # number of trivial fixed points
        n_trivial_fixed_pts = count_trivial_fixed_pts(trajectory, atol)
        n_fixed_pts = count_fixed_pts(trajectory, atol)

        print(n_fixed_pts, n_trivial_fixed_pts)
        # Compute the number of nontrivial fixed points from their difference
        return (n_fixed_pts - n_trivial_fixed_pts)

    plt.title(
        "The proportion of neurons at non-trivial fixed points with coupling strength"
    )
    plt.xlabel("$g$, the coupling strength")
    plt.ylabel("Fraction of neurons at non-trivial fixed points")
    plot_with_g(gs, measure, normalize=True, **kwargs)


def plot_cycles_with_g(gs: Sequence,
                       atol: float = 1e-3,
                       max_n_steps: int = 10000,
                       **kwargs):
    """
    Plot the fraction of dofs which settle to an oscillatory cycle
    (i.e. neither noisy behavior nor fixed points) as a function of
    the coupling strength, $g$.

    :param gs: The list of coupling strengths for which to sample
        corresponding lyapunov exponents.
    :param atol: This is the absolute numerical tolerance that
        determines whether a subtrajectory has settled into a cycle.
    :param max_n_steps: The maximum number of steps to use to compute
        cycles.  This computes cycles by looking at the
        autocorrelation, and if the trajectories are too long, this
        function will take unreasonably long to return.
    :param kwargs: See `plot_with_g`.
    """

    measure = lambda trajectory, _: count_cycles(trajectory, atol,
                                                 max_n_steps)

    plt.title(
        "The proportion of neurons in a regular cycle with coupling strength")
    plt.xlabel("$g$, the coupling strength")
    plt.ylabel("Fraction of neurons in cycles")
    plot_with_g(gs, measure, normalize=True, **kwargs)


def plot_participation_ratio_with_g(gs: Sequence,
                                    max_n_steps: int = 10000,
                                    **kwargs):
    """
    Plot the maximum lyapunov exponent as a function of the coupling
    strength, $g$.

    :param gs: The list of coupling strengths for which to sample
        corresponding lyapunov exponents.
    :param max_n_steps: The number of samples (drawn from the end of
        the trajectory) over which to compute a PCA for the
        participation ratio.  Very long trajectory lengths may be more
        than the computer can reasonably handle.
    :param kwargs: See `plot_with_g`.
    """

    measure = lambda trajectory, _: participation_ratio(
        trajectory.T, max_n_steps=max_n_steps)

    plt.title("Relative Participation ratio, $D/N$")
    plt.xlabel("$g$, the coupling strength")
    plt.ylabel("Participation ratio, $D_{PCA}$")
    plot_with_g(gs, measure, normalize=True, **kwargs)

def pretty_histogram(seq: Sequence, bins: int=100):
    plt.hist(seq, bins=bins)
    plt.axvline(x=np.mean(seq))
