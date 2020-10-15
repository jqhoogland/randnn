"""

Contains code to implement the transfer operator approach

Author: Jesse Hoogland
Year: 2020

"""
import logging
from dataclasses import dataclass
from typing import Union,List, NewType

import numpy as np
from scipy import linalg
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tqdm.notebook import tqdm
from tqdm.contrib import tenumerate

from .utils import eigsort, normalize_rows, svd_whiten

LabelingMethod = NewType("LabelingMethod", Union["kmeans", "uniform"])

@dataclass
class TransferOperator(object):
    """Class for implementing the transfer operator approach"""
    trans_matrix: np.ndarray = None
    invariant_dist: np.ndarray = None
    minimum_x: float = 0.
    maximum_x: float = 0.
    labeling_method:LabelingMethod  = "kmeans"


    @staticmethod
    def get_uniform_labels(time_series: np.ndarray,
                           n_clusters: int,
                           verbose: bool = False) -> np.ndarray:
        """
        Uniform labeling is one of two possible labeling methods, the other being kmeans clustering.

        Uniform labeling partitions the phase space into uniformly-spaced boxes.
        Currently, this only works for 1d phase spaces.

        TODO: Extend this to higher dimensions. Even then, it will only work for
        low numbers of dimensions since the number of partitions scales rapidly.
        """

        # To uniformly partition the points in a trajectory, we have to determine the boundaries.
        minimum_x = np.amin(time_series, axis=0)
        maximum_x = np.amax(time_series, axis=0)

        n_bins_per_dim = time_series.shape[1]

        step = (maximum_x - minimum_x) / np.power(n_clusters,
                                                  (1. / n_bins_per_dim))

        labels = np.zeros((time_series.shape[0], ), dtype=np.int32)

        def make_label(bucket_labels):
            # The aim of this function is to turn label of the timeseries along each axis into a unique collective label.

            if not np.isscalar(bucket_labels):
                label = int(bucket_labels[0])
            else:
                label = int(bucket_labels)

            #     for (i, label) in enumerate(bucket_labels):
            #         label += (n_bins_per_dim ** i) * label

            #     bucket_labels = label

            if label >= n_clusters - 0.5:
                label = int(n_clusters - 1)

            return label

        if verbose:
            for i in tqdm(range(time_series.shape[0]),
                          desc="Generating labels"):
                state = time_series[i, :]
                labels[i] = make_label(np.floor_divide(state - minimum_x,
                                                       step))
        else:
            for i in range(time_series.shape[0]):
                state = time_series[i, :]
                labels[i] = make_label(np.floor_divide(state - minimum_x,
                                                       step))

        return labels

    @staticmethod
    def get_kmeans_labels(time_series: np.ndarray,
                          n_clusters: int) -> np.ndarray:
        """
        Kmeans labeling is one of two possible labeling methods, the other being uniform labeling.

        Kmeans clusters the phase space into maximally explicative Voronoi cells.

        This is the better choice in higher dimensions because it is more efficient.
        However, at lower dimensions, uniform labeling is preferred because it is more visually interpretable
        (or at least easier to turn into something visually interpretable).

        TODO: Save the kmeans fit so that we don't have to do this every time all over again.
        """
        return KMeans(n_clusters).fit(time_series).labels_

    def get_ulam_galerkin_labels(self,
                                 time_series: np.ndarray,
                                 n_clusters: int,
                                 verbose: bool = False) -> np.ndarray:
        """
        Produces the cluster identity labels for each entry in a time series
        `time_series` using  `n_clusters`.
        Decides what kind of labels to produce according to `self.labeling_method`

        :param time_series: (np.ndarray of shape [t, d])
        :param n_clusters: (int)

        :returns cluster_identities: (np.ndarray of shape [t], dtype="int32", with values chosen from `range(0, n_clusters)`)
        representing the cluster identities of each time entry.
        """

        if self.labeling_method == "kmeans":
            return self.get_kmeans_labels(time_series, n_clusters)
        elif self.labeling_method == "uniform":
            return self.get_uniform_labels(time_series, n_clusters, verbose)
        else:
            raise ValueError(
                f"labeling_method must be one of 'kmeans', 'unifom', but is {self.labeling_method}"
            )

    @staticmethod
    def _get_trans_matrix_unnormalized(cluster_labels: np.ndarray,
                                       n_clusters: int,
                                       n_future_timesteps: int,
                                       verbose: bool = False) -> np.ndarray:
        """
        We approximate the transfer operator as a finite-rank matrix $P_{i,j}(\tau, N)$, where
        entry $(i,j)$ counts the number of transitions between cluster $i$ and $j$ in
        a time window $\tau$.
        Here, $N$ is the number of clusters, i.e. $P$ will have shape $(N, N)$.
        """

        # We have discretized time, so $\tau$ becomes a discrete number of steps, `n_future_timesteps`.
        # This has to be at least 1, else we would not be able to measure any transitions.
        assert n_future_timesteps >= 1 and isinstance(n_future_timesteps, int), "`n_future_timesteps` must be an integer greater than 1"

        # For efficiency, we use sparse matrices
        trans_matrix = sp.lil_matrix((n_clusters, n_clusters))

        cluster_labels_range = range(cluster_labels.size - n_future_timesteps - 1 )

        # For a verbose readout, we log our progress along the trajectory
        if verbose:
            cluster_labels_range = tqdm(cluster_labels_range,
                                        desc="Creating the transfer matrix")

        for i in cluster_labels_range:
            future_timestep = i + n_future_timesteps
            trans_matrix[cluster_labels[i], cluster_labels[future_timestep]] += 1

            # for future_timestep in range(
            #         i + 1,
            #         min(cluster_labels.size, i + n_future_timesteps + 1)):
            #     # If a state in one cluster i ends up in another cluster j within n_timesteps steps,
            #     # Then, we add one to the trans matrix for the trans i -> j
            #     trans_matrix[cluster_labels[i],
            #                  cluster_labels[future_timestep]] += 1

        return trans_matrix.toarray()

    def get_trans_matrix(self,
                         time_series: np.ndarray,
                         n_clusters: int,
                         n_future_timesteps: int = 1) -> np.ndarray:

        # 1. Convert the time_series into a series of cluster labels
        cluster_labels = self.get_ulam_galerkin_labels(time_series, n_clusters)

        # 2. Compute the unnormalized trans_matrix from this series of cluster labels
        trans_matrix_unnormalized = self._get_trans_matrix_unnormalized(
            cluster_labels, n_clusters, n_future_timesteps)

        # 3. Normalize and return
        trans_matrix = normalize_rows(trans_matrix_unnormalized)

        return trans_matrix

    def get_t_imp(self, time_series: np.ndarray, eigval_idxs: List[int],
                  n_clusters: int, tau: int,
                  transition_time: np.float64) -> np.float64:

        trans_matrix = self.get_trans_matrix(time_series,
                                             n_clusters=n_clusters,
                                             n_future_timesteps=tau)

        n_eigvals = max(eigval_idxs) + 1

        eig_method = "sp"
        if (n_eigvals >= trans_matrix.shape[0] -1):
            eig_method = "np"

        eigvals = eigsort(trans_matrix.T, max(eigval_idxs) + 1, which="LM", eig_method=eig_method)[0]

        eigval = eigvals[eigval_idxs]
        t_imp = -transition_time * tau / (np.log(np.abs(eigval)))

        return t_imp

    @staticmethod
    def _get_inv_trans_matrix(trans_matrix: np.ndarray,
                              invariant_dist: np.ndarray) -> np.ndarray:
        return (sp.diags(1.0 / invariant_dist) * trans_matrix.T *
                sp.diags(invariant_dist))

    def get_reversible_trans_matrix(
        self,
        trans_matrix: np.ndarray,
        invariant_dist: np.ndarray,
    ) -> np.ndarray:
        inv_trans_matrix = self._get_inv_trans_matrix(trans_matrix,
                                                      invariant_dist)
        reversible_trans_matrix = (trans_matrix + inv_trans_matrix) / 2.0

        return reversible_trans_matrix

    def fit_reversible_trans_matrix(self):
        if self.trans_matrix is None or self.invariant_dist is None:
            raise AttributeError(
                "Transfer Operator must be trained before accessing reversible transfer operator."
            )

        self.reversible_trans_matrix = self.get_reversible_trans_matrix(
            self.trans_matrix, self.invariant_dist)

    @staticmethod
    def get_invariant_dist(trans_matrix):
        """
        :returns invariant_dist: (np.ndarray of shape [trans_matrix.shape[0],])
        i.e. a row vector with values $[1, 1, 1,...,1] / \sqrt{N}$
        """

        # (But then normalized)
        transfer_eigvals, transfer_eigvecs = np.linalg.eig(trans_matrix.T)

        # We want the largest eigenvector
        idx = transfer_eigvals.argsort()[::-1]
        transfer_eigvecs = transfer_eigvecs[:, idx]

        return np.abs(transfer_eigvecs[:, 0])

    @staticmethod  # NOTE: This has been tested
    def get_partitioning_entropy(trans_matrix, invariant_dist):
        """
        Calculates the entropy rate of a given partitioning (of $$N$$ sites) through

        $$ h(N)=-\sum_{i,j=1}^N \pi_i P_{ij}(dt, N)\log P_{ij}(dt,N),$$
        where $$P$$ is the transfer matrix and $$\pi$$ is its first left eigenvector, the invariant distribution.

        """
        logging.debug(
            "------------------------------------------------------------")
        logging.debug(
            "_get_partitioning_entropy:INPUTS:trans_matrix:SHAPE:{}".format(
                trans_matrix.shape))
        logging.debug(
            "_get_partitioning_entropy:INPUTS:invariant_dist:SHAPE:{}".format(
                invariant_dist.shape))

        if type(trans_matrix) != np.ndarray:
            trans_matrix = trans_matrix.toarray()

        invariant_dist = invariant_dist.reshape([1, -1])

        trans_matrix_ones = trans_matrix.copy()
        trans_matrix_ones[trans_matrix ==
                          0] = 1  # To avoid infinities in np.log

        logging.log(
            5,
            "\ntrans_matrix: {}, \ntrans_matrix_ones:{}\ninvariant_dist: {}".
            format(trans_matrix, trans_matrix_ones, invariant_dist))
        logging.log(
            5, "\nentropies {}".format(
                -invariant_dist * (trans_matrix * np.log(trans_matrix_ones))))

        return -np.sum(invariant_dist *
                       (trans_matrix * np.log(trans_matrix_ones)))


# ------------------------------------------------------------

# Phase Space Reconstruction (as set out in Costa 2020), works at the level of transfer matrices

# ------------------------------------------------------------

@dataclass
class PhaseSpaceReconstructor(TransferOperator):
    n_delays: int = 0
    embedding_dim: int = 1

    def get_max_entropy_partitioning(self,
                                     time_series,
                                     clusters_range=np.arange(100, 2500, 100),
                                     n_future_timesteps=1):
        """
        Determines the maximum entropy of ``time_series`` over partitions of varying
        numbers of cells, specified in ``clusters_range``

        :param time_series: (np.ndarray of shape [t, X]), note that the spatial dimension can be any number.
        :param clusters_range: (list-like) a selection of clusters sizes to choose from.
        defaults to ``np.arange(100,2500,100)``

        :returns max_entropy: the maximum entropy over all partitions considered.
        :returns trans_matrix: the transfer matrix produced from the corresponding partitioning,
        :returns invariant_dist: the invariant distribution produced from the corresponding partitioning,

        """
        logging.debug(
            "------------------------------------------------------------")
        logging.debug(
            "get_max_entropy_partitioning:INPUTS:time_series:SHAPE:{}".format(
                time_series.shape))
        logging.debug(
            "get_max_entropy_partitioning:INPUTS:clusters_range:SHAPE:{}".
            format(clusters_range.shape))
        logging.debug(
            "get_max_entropy_partitioning:INPUTS:n_future_timesteps:VALUE:{}".
            format(n_future_timesteps))

        entropies = np.zeros(clusters_range.size)
        trans_matrices = []
        invariant_dists = []

        # For a range of candidate ``N`` numbers of clusters, we calcualte the entropy
        for i in range(
                clusters_range.size
        ):  #tqdm(range(clusters_range.size), desc="Determining maximum entropy over partitionings"):
            n_clusters = clusters_range[i]
            trans_matrices.append(
                self.get_trans_matrix(time_series, n_clusters,
                                      n_future_timesteps))
            invariant_dists.append(self.get_invariant_dist(trans_matrices[i]))

            entropies[i] = self.get_partitioning_entropy(
                trans_matrices[i], invariant_dists[i])

            logging.log(
                15, "Entropy for {} partitions determined to be {}".format(
                    n_clusters, entropies[i]))

        max_idx = np.argmax(entropies)

        # We return the maximum entropy (and number of clusters) of this set.
        return (entropies[max_idx], trans_matrices[max_idx],
                invariant_dists[max_idx]
                )  # TODO: Use the second returned argument somewhere

    @staticmethod
    def get_first_nondecreasing_element_idx(arr, min_idx=1):
        # Choose the first delay for which the entropy is no longer decreasing.
        # If there is no such element, returns the index of the last element
        delta_arr = (arr - np.roll(arr, 1))[1:]

        k = min_idx
        while k < delta_arr.size:
            if delta_arr[k] >= 0:
                break

            k += 1

        return (
            k
        )  # because we could only measure deltas starting at the first position in ``arr``

    def get_optimal_n_delays(
        self,
        time_series,
        delay_range=np.arange(1, 50),
        clusters_range=np.arange(100, 2500, 100),
        n_future_timesteps=1,
        tol=1e-1,
    ):
        """
        Determines the optimal number of delays to use in creating a delay embedding.
        The optimal element is the first number of delays for which the maximum entropy of the ``time_series`` is no longer decreasing.
        The maximum entropy of the ``time_series`` is the maximum entropy for all possible partitionings with sizes chosen from ``clusters_range``


        :param time_series: (np.ndarray of shape [t, d])
        :param delay_range: (list-like) a selection of embedding dimensions to choose from.
        defaults to ``np.arange(1,50)``
        :param clusters_range: see ``self.get_max_entropy_partitioning``

        :returns optimal_n_delays: the optimal choice of delays from the preset range ``delay_range``
        """
        logging.debug(
            "------------------------------------------------------------")
        logging.debug(
            "------------------------------------------------------------")
        logging.info("Optimizing the number of delays ``n_delays``.")

        logging.debug(
            "get_optimal_n_delays:INPUTS:time_series:SHAPE:{}".format(
                time_series.shape))
        logging.debug(
            "get_optimal_n_delays:INPUTS:delay_range:SHAPE:{}".format(
                delay_range.shape))
        logging.debug(
            "get_optimal_n_delays:INPUTS:clusters_range:SHAPE:{}".format(
                clusters_range.shape))
        logging.debug(
            "get_optimal_n_delays:INPUTS:n_future_timesteps:VALUE:{}".format(
                n_future_timesteps))

        n_delays = delay_range[-1]
        trans_matrix, invariant_dist = None, None

        entropies = np.zeros(delay_range.size)

        for i in tqdm(range(delay_range.size),
                      desc="Determining optimal number of delays:"):

            n_delays = delay_range[i]

            delay_embedded_series = self._get_delay_embedding(
                time_series, n_delays)

            entropies[
                i], _trans_matrix, _invariant_dist = self.get_max_entropy_partitioning(
                    delay_embedded_series,
                    clusters_range=clusters_range,
                    n_future_timesteps=n_future_timesteps,
                )

            logging.debug(
                "------------------------------------------------------------")
            logging.info(
                "Entropy for sequence of {} delays determined to be {}".format(
                    n_delays, entropies[i]))
            logging.debug(
                "------------------------------------------------------------")

            if i > 0 and entropies[i] > entropies[i - 1] - tol:
                # we have encountered an increasing entropy and can break out of the for loop
                n_delays = delay_range[i - 1]
                break

            trans_matrix = _trans_matrix
            invariant_dist = _invariant_dist

        logging.info(
            "Optimal number of delays ``n_delay`` determined to be {}".format(
                n_delays))
        return n_delays, trans_matrix, invariant_dist

    def get_optimal_embedding_dim(
        self,
        delay_embedded_series,
        embedding_dim_range=None,
        clusters_range=np.arange(100, 1000, 100),
        n_future_timesteps=1,
        tol=1e-3,
    ):
        """
        Assumes the optimal ``self.n_delays`` has already been determined.

        Determines the optimal number of components to keep during the dimensional reduction step.
        The optimal element is the first embedding dimension for which the maximum entropy of the ``time_series`` is no longer decreasing.
        The maximum entropy of the ``time_series`` is the maximum entropy for all possible partitionings with sizes chosen from ``clusters_range`` with a given embedding dimensionality

        :param time_series: (np.ndarray of shape [self.n_delays * d, t])
        :param embedding_dim_range: (``None`` or list-like), a selection of embedding dimensions to choose from.
        Defaults to None, in which case all possible numbers of dimensions are considered (between 1 and ``d``)
        :param clusters_range: see ``self.get_max_entropy_partitioning``

        :returns optimal_embedding_dim: (int) the optimal choice of embedding dimension from the preset range ``embedding_dim_range``
        """
        logging.debug(
            "------------------------------------------------------------")
        logging.debug(
            "------------------------------------------------------------")

        logging.info("Optimizing the embedding dimension ``embedding_dim``.")

        if embedding_dim_range is None:
            embedding_dim_range = np.arange(1, delay_embedded_series.shape[1])

        logging.debug(
            "get_optimal_embedding_dim:INPUTS:delay_embedded_series:SHAPE:{}".
            format(delay_embedded_series.shape))
        logging.debug(
            "get_optimal_embedding_dim:INPUTS:embedding_dim_range:SHAPE:{}".
            format(embedding_dim_range.shape))
        logging.debug(
            "get_optimal_embedding_dim:INPUTS:clusters_range:SHAPE:{}".format(
                clusters_range.shape))
        logging.debug(
            "get_optimal_embedding_dim:INPUTS:n_future_timesteps:VALUE:{}".
            format(n_future_timesteps))

        embedding_dim = embedding_dim_range[-1]
        entropies = np.zeros(embedding_dim_range.size)

        # Whiten the delay embedded time series
        delay_embedded_series_whitened = svd_whiten(delay_embedded_series)

        for i in tqdm(range(embedding_dim_range.size),
                      desc="Determining optimal embedding dimension:"):
            embedding_dim = embedding_dim_range[i]

            if embedding_dim > delay_embedded_series.shape[1]:
                # We do not want to project to a higher dimensional space
                if (i > 0):
                    embedding_dim = embedding_dim_range[i - 1]
                else:
                    raise ValueError(
                        "Insuitable embedding_dim_range provided. Must include values (in increasing order) less than {}. Value is {}"
                        .format(delay_embedded_series.shape[1],
                                embedding_dim_range))
                break

            # For each embedding_dim, determine the maximum entropy over all cluster sizes in ``clusters_range``
            # for the delay series resulting from keeping only ``embedding_dim`` many components of the delay series.
            entropies[i] = self.get_max_entropy_partitioning(
                delay_embedded_series_whitened[:, :embedding_dim],
                clusters_range=clusters_range,
                n_future_timesteps=n_future_timesteps,
            )[0]

            if i > 0 and entropies[i] > entropies[i - 1] - tol:
                # we have encountered an increasing entropy and can break out of the for loop
                embedding_dim = embedding_dim_range[i - 1]
                break

            logging.info(
                "Entropy for sequence of embedding dimension {} determined to be {}"
                .format(embedding_dim, entropies[i]))

        logging.info(
            "Optimal embedding dimension ``embedding_dim`` determined to be {}"
            .format(embedding_dim))

        return embedding_dim

    def fit_trans_matrix(
        self,
        time_series,
        delay_range=np.arange(1, 50),
        delay_clusters_range=np.arange(100, 1000, 100),
        delay_n_future_timesteps=1,
    ):
        if self.n_delays is None:
            self.n_delays, self.trans_matrix, self.invariant_dist = self.get_optimal_n_delays(
                time_series,
                delay_range=delay_range,
                clusters_range=delay_clusters_range,
                n_future_timesteps=delay_n_future_timesteps)
        else:
            _, self.trans_matrix, self.invariant_dist = self.get_max_entropy_partitioning(
                time_series,
                clusters_range=delay_clusters_range,
                n_future_timesteps=delay_n_future_timesteps)

        self.fit_reversible_trans_matrix()
        return self.reversible_trans_matrix


    @staticmethod
    def _get_delay_embedding(time_series: np.ndarray,
                             n_delays: int) -> np.ndarray:
        """
        A helper method which produces a delay-embedded time series from ``time_series`` with ``n_delays``

        :param time_series: (shape [t, d])
        :param n_delays: number of delays to embed

        :returns delay_embedded_series: (shape [t - n_delays, d * n_delays])

        """

        logging.debug(
            "------------------------------------------------------------")
        logging.debug(
            "_get_delay_embedding:INPUTS:time_series:SHAPE:{}".format(
                time_series.shape))
        logging.debug(
            "_get_delay_embedding:INPUTS:n_delays:VALUE:{}".format(n_delays))

        delay_embedding = time_series

        if n_delays == 0:
            return time_series

        for i in range(1, n_delays + 1):
            # Concatenatenate a [t, d]-shape array along its secondary axis
            # with a delayed version of itself.
            # (i.e. the same matrix rolled backwards along its principal axis)
            # Then, we cut off the last row, since we have no information about the
            # delays after our trajectory samples.

            delay_embedding = np.concatenate(
                [delay_embedding, np.roll(time_series, -i, 0)], axis=1)

        delay_embedding = delay_embedding[:delay_embedding.shape[0] -
                                          n_delays, :]

        logging.debug(
            "------------------------------------------------------------")
        logging.debug(
            "_get_delay_embedding:OUTPUT:delay_embedding:SHAPE:{}".format(
                delay_embedding.shape))

        return delay_embedding

    def get_delay_embedding(self, time_series: np.ndarray) -> np.ndarray:
        """
        A wrapper for ``self._get_delay_embedding`` which uses ``self.n_delays`` (which must already be optimized/fitted) as the number of delays.
        """
        if self.n_delays is None:
            raise AttributeError(
                "You must first fit or assign the attribute ``n_delays``.")

        return self._get_delay_embedding(time_series, self.n_delays)


    def _fit(
        self,
        time_series,
        delay_range=np.arange(1, 50),
        embedding_dim_range=None,
        delay_clusters_range=np.arange(100, 1000, 100),
        dim_clusters_range=np.arange(100, 1000, 100),
        delay_n_future_timesteps=1,
        dim_n_future_timesteps=1,
        downsample=None,
    ):
        """
        Helper function to be used with either ``self.fit`` or ``self.fit_transform``
        This function always returns the transformed ``time_series``

        :param time_series: (np.ndarray of shape [t, d])
        :param embedding_dim_range: see ``self.get_optimal_embedding_dim``

        :returns reconstructed_series: (np.ndarray of shape [t, self.embedding_dim])
        """

        if not (downsample is None):
            time_series = self.downsample(time_series, downsample)

        # Step 1: Determine the optimal delay length if not already fit

        self.minimum_x = np.amin(time_series, axis=0)
        self.maximum_x = np.amax(time_series, axis=0)

        self.fit_trans_matrix(
            time_series,
            delay_range=delay_range,
            delay_clusters_range=delay_clusters_range,
            delay_n_future_timesteps=delay_n_future_timesteps)

        # Step 2: Create the appropriate delay embedding
        delay_embedding = self.get_delay_embedding(time_series)

        # Step 2: Determine the optimal embedding dimension if not already determined.
        if self.embedding_dim is None:
            self.embedding_dim = self.get_optimal_embedding_dim(
                delay_embedding,
                embedding_dim_range=
                embedding_dim_range,  # if this contains dimensions higher than the n_delays, this is automatically adjusted
                clusters_range=dim_clusters_range,
                n_future_timesteps=dim_n_future_timesteps,
            )

        # Step 3: Perform a PCA and transform (and whiten) the delay embedding,
        # keeping only the self.embedding_dim many components
        pca = PCA(self.embedding_dim, whiten=True)
        self.pca = pca.fit(delay_embedding)
        reconstructed_series = self.pca.transform(delay_embedding)

        # Step 4: Return the dimensionally-reduced reconstructed embedding
        return reconstructed_series

    def fit(
        self,
        time_series,
        delay_range=np.arange(1, 50),
        embedding_dim_range=None,
        delay_clusters_range=np.arange(100, 1000, 100),
        dim_clusters_range=np.arange(100, 1000, 100),
        delay_n_future_timesteps=1,
        dim_n_future_timesteps=1,
        downsample=None,
    ):
        """
        Function which fits this TransferOperator object to the ``time_series``
        and returns a reference to itself.

        :param time_series: (np.ndarray of shape [t, d])
        """

        self._fit(time_series,
                  delay_range=delay_range,
                  embedding_dim_range=embedding_dim_range,
                  delay_clusters_range=delay_clusters_range,
                  dim_clusters_range=dim_clusters_range,
                  delay_n_future_timesteps=delay_n_future_timesteps,
                  dim_n_future_timesteps=dim_n_future_timesteps,
                  downsample=downsample)
        return self

    def transform(self, time_series):
        """
        Assumes that this object has already been fitted using ``fit``.

        :param time_series: (np.ndarray of shape [t, d])

        First, embeds the ``time_series`` in a delay-embeded space of ``self.n_delays`` delays,
        i.e. to a np.ndarray of shape [self.n_delays * d, t]

        Then, performs dimensional reduction by performing SVD and keeping only the ``self.embedding_dim`` many components.

        :returns reconstructed_series: (np.ndarray of shape [t, self.embedding_dim])
        """

        delay_embedding = get_delay_embedding(time_series, n_delays)

        if self.pca is None:
            raise AttributeError(
                "You must first fit the transfer operator before using transform."
            )

        return self.pca.transform(delay_embedding)

    def fit_transform(
        self,
        time_series,
        delay_range=np.arange(1, 50),
        embedding_dim_range=None,
        delay_clusters_range=np.arange(100, 1000, 100),
        dim_clusters_range=np.arange(100, 1000, 100),
        delay_n_future_timesteps=1,
        dim_n_future_timesteps=1,
        downsample=None,
    ):
        """
        Function which fits this TransferOperator object to the ``time_series``
        and returns the transformed ``time_series``

        :param time_series: (np.ndarray of shape [t, d])

        :returns reconstructed_series: (np.ndarray of shape [t, self.embedding_dim])
        """

        return self._fit(time_series,
                         delay_range=delay_range,
                         embedding_dim_range=embedding_dim_range,
                         delay_clusters_range=delay_clusters_range,
                         dim_clusters_range=dim_clusters_range,
                         delay_n_future_timesteps=delay_n_future_timesteps,
                         dim_n_future_timesteps=dim_n_future_timesteps,
                         downsample=downsample)



# pytest


def test_get_delay_embedding_1():
    transfer_operator = TransferOperator(n_delays=3)
    assert (transfer_operator.get_delay_embedding(
        np.arange(10).reshape([10, 1])) == np.array([
            np.arange(0, 7),
            np.arange(1, 8),
            np.arange(2, 9),
            np.arange(3, 10)
        ]).T).all()
    assert (transfer_operator.get_delay_embedding(
        np.concatenate(
            [np.arange(10).reshape([10, 1]),
             np.arange(10).reshape([10, 1])],
            axis=1)) == np.array([
                np.arange(0, 7),
                np.arange(0, 7),
                np.arange(1, 8),
                np.arange(1, 8),
                np.arange(2, 9),
                np.arange(2, 9),
                np.arange(3, 10),
                np.arange(3, 10),
            ]).T).all()


def test_get_trans_matrix_unnormalized_1():
    transfer_operator = TransferOperator()

    assert np.isclose(
        transfer_operator._get_trans_matrix_unnormalized(
            np.array([0, 0, 1, 2, 1, 0, 2, 1]), 3, 1).toarray(),
        np.array([[1, 1, 1], [1, 0, 1], [0, 2, 0]]),
    ).all()


def test_get_trans_matrix_unnormalized_2():
    transfer_operator = TransferOperator()

    assert np.isclose(
        transfer_operator._get_trans_matrix_unnormalized(
            np.array([0, 0, 1, 2, 1, 0, 2, 1]), 3, 2).toarray(),
        np.array([[1, 3, 2], [1, 1, 2], [1, 2, 0]]),
    ).all()


def test_get_invariant_dist():
    transfer_operator = TransferOperator()

    assert np.isclose(
        transfer_operator.get_invariant_dist(
            np.array([[0.5, 0.1, 0.4], [0, 0.5, 0.5], [0.25, 0.25, 0.5]])),
        np.array([1, 1, 1]) / np.sqrt(3),
    ).all()


def test_get_partitioning_entropy():
    transfer_operator = TransferOperator()

    partitioning_entropy = transfer_operator._get_partitioning_entropy(
        np.array([[0.5, 0.1, 0.4], [0, 0.5, 0.5], [0.25, 0.25, 0.5]]),
        np.array([1, 1, 1]) / np.sqrt(3),
    )
    print(partitioning_entropy)
    assert np.isclose(
        partitioning_entropy,
        1.545114226461558,
    )


def test_get_first_nondecreasing_element_idx_1():
    transfer_operator = TransferOperator()

    assert transfer_operator.get_first_nondecreasing_element_idx(
        np.array([1, .5, .25, .2, .15, .05, .06, .03, 0, .3])) == 5


def test_get_first_nondecreasing_element_idx_2():
    transfer_operator = TransferOperator()

    assert transfer_operator.get_first_nondecreasing_element_idx(
        np.array([1, .5, .25, .2, .15, .05, .06, .03, 0, .3]), 6) == 8


def test_get_first_nondecreasing_element_idx_3():
    transfer_operator = TransferOperator()

    assert transfer_operator.get_first_nondecreasing_element_idx(
        np.array([1, .5, .25, .2, .15, .05, .06, .03, 0, .3]), 8) == 8


def test_uniform_clusters_1():
    transfer_operator = TransferOperator(labeling_method="uniform")

    assert np.isclose(
        transfer_operator.get_ulam_galerkin_labels(np.arange(0, 11), 10),
        np.arange(0, 11)).all()


def test_uniform_clusters_2():
    transfer_operator = TransferOperator(labeling_method="uniform")

    assert np.isclose(
        transfer_operator.get_ulam_galerkin_labels(np.arange(0, 10.5, 0.5),
                                                   10),
        np.floor_divide(np.arange(0, 21), 2)).all()


# def test_time_inverse_transfer_operator():
#     trans_matrix = sp.csr_matrix(
#         np.array([[0, 0.5, 0.5], [0.2, 0.5, 0.3], [0.1, 0.05, 0.85]])
#     )
#     transfer_operator = TransferOperator(n_delays=1)
#     assert np.isclose(
#         np.array(
#             transfer_operator.get_revers_transfer_operator(
#                 trans_matrix
#             ).todense(),
#             dtype="float64",
#         ),
#         (
#             np.array(
#                 [[0, 0.2, 0.1], [0.5, 0.5, 0.05], [0.5, 0.3, 0.85]], dtype="float64"
#             )
#         ),
#     ).all()
