import warnings
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import ward_tree, _hierarchical
from sklearn.cluster._feature_agglomeration import AgglomerationTransform
from sklearn.cluster.hierarchical import _complete_linkage, _average_linkage, _single_linkage, _hc_cut
from sklearn.utils import check_array
from sklearn.utils.validation import check_memory

import numpy as np

_TREE_BUILDERS = dict(
    ward=ward_tree,
    complete=_complete_linkage,
    average=_average_linkage,
    single=_single_linkage)


class AgglomerativeClustering(BaseEstimator, ClusterMixin):
    """
    Agglomerative Clustering

    Recursively merges the pair of clusters that minimally increases
    a given linkage distance.

    Read more in the :ref:`User Guide <hierarchical_clustering>`.

    Parameters
    ----------
    n_clusters : int or None, optional (default=2)
        The number of clusters to find. It must be ``None`` if
        ``distance_threshold`` is not ``None``.

    affinity : string or callable, default: "euclidean"
        Metric used to compute the linkage. Can be "euclidean", "l1", "l2",
        "manhattan", "cosine", or "precomputed".
        If linkage is "ward", only "euclidean" is accepted.
        If "precomputed", a distance matrix (instead of a similarity matrix)
        is needed as input for the fit method.

    memory : None, str or object with the joblib.Memory interface, optional
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    connectivity : array-like or callable, optional
        Connectivity matrix. Defines for each sample the neighboring
        samples following a given structure of the data.
        This can be a connectivity matrix itself or a callable that transforms
        the data into a connectivity matrix, such as derived from
        kneighbors_graph. Default is None, i.e, the
        hierarchical clustering algorithm is unstructured.

    compute_full_tree : bool or 'auto' (optional)
        Stop early the construction of the tree at n_clusters. This is
        useful to decrease computation time if the number of clusters is
        not small compared to the number of samples. This option is
        useful only when specifying a connectivity matrix. Note also that
        when varying the number of clusters and using caching, it may
        be advantageous to compute the full tree. It must be ``True`` if
        ``distance_threshold`` is not ``None``.

    linkage : {"ward", "complete", "average", "single"}, optional \
            (default="ward")
        Which linkage criterion to use. The linkage criterion determines which
        distance to use between sets of observation. The algorithm will merge
        the pairs of cluster that minimize this criterion.

        - ward minimizes the variance of the clusters being merged.
        - average uses the average of the distances of each observation of
          the two sets.
        - complete or maximum linkage uses the maximum distances between
          all observations of the two sets.
        - single uses the minimum of the distances between all observations
          of the two sets.

    pooling_func : callable, default='deprecated'
        Ignored.

        .. deprecated:: 0.20
            ``pooling_func`` has been deprecated in 0.20 and will be removed
            in 0.22.

    distance_threshold : float, optional (default=None)
        The linkage distance threshold above which, clusters will not be
        merged. If not ``None``, ``n_clusters`` must be ``None`` and
        ``compute_full_tree`` must be ``True``.

        .. versionadded:: 0.21

    Attributes
    ----------
    n_clusters_ : int
        The number of clusters found by the algorithm. If
        ``distance_threshold=None``, it will be equal to the given
        ``n_clusters``.

    labels_ : array [n_samples]
        cluster labels for each point

    n_leaves_ : int
        Number of leaves in the hierarchical tree.

    n_connected_components_ : int
        The estimated number of connected components in the graph.

    children_ : array-like, shape (n_samples-1, 2)
        The children of each non-leaf node. Values less than `n_samples`
        correspond to leaves of the tree which are the original samples.
        A node `i` greater than or equal to `n_samples` is a non-leaf
        node and has children `children_[i - n_samples]`. Alternatively
        at the i-th iteration, children[i][0] and children[i][1]
        are merged to form node `n_samples + i`

    Examples
    --------
    >>> from sklearn.cluster import AgglomerativeClustering
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 4], [4, 0]])
    >>> clustering = AgglomerativeClustering().fit(X)
    >>> clustering # doctest: +NORMALIZE_WHITESPACE
    AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto',
                            connectivity=None, distance_threshold=None,
                            linkage='ward', memory=None, n_clusters=2,
                            pooling_func='deprecated')
    >>> clustering.labels_
    array([1, 1, 1, 0, 0, 0])

    """

    def __init__(self, n_clusters=2, affinity="euclidean",
                 memory=None,
                 connectivity=None, compute_full_tree='auto',
                 linkage='ward', pooling_func='deprecated',
                 distance_threshold=None):
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold
        self.memory = memory
        self.connectivity = connectivity
        self.compute_full_tree = compute_full_tree
        self.linkage = linkage
        self.affinity = affinity
        self.pooling_func = pooling_func

    @property
    def n_components_(self):
        return self.n_connected_components_

    def fit(self, X, y=None):
        """Fit the hierarchical clustering on the data

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training data. Shape [n_samples, n_features], or [n_samples,
            n_samples] if affinity=='precomputed'.

        y : Ignored

        Returns
        -------
        self
        """
        if (self.pooling_func != 'deprecated' and
                not isinstance(self, AgglomerationTransform)):
            warnings.warn('Agglomerative "pooling_func" parameter is not used.'
                          ' It has been deprecated in version 0.20 and will be'
                          'removed in 0.22', DeprecationWarning)
        X = check_array(X, ensure_min_samples=2, estimator=self)
        memory = check_memory(self.memory)

        if self.n_clusters is not None and self.n_clusters <= 0:
            raise ValueError("n_clusters should be an integer greater than 0."
                             " %s was provided." % str(self.n_clusters))

        if not ((self.n_clusters is None) ^ (self.distance_threshold is None)):
            raise ValueError("Exactly one of n_clusters and "
                             "distance_threshold has to be set, and the other "
                             "needs to be None.")

        if (self.distance_threshold is not None
                and not self.compute_full_tree):
            raise ValueError("compute_full_tree must be True if "
                             "distance_threshold is set.")

        if self.linkage == "ward" and self.affinity != "euclidean":
            raise ValueError("%s was provided as affinity. Ward can only "
                             "work with euclidean distances." %
                             (self.affinity, ))

        if self.linkage not in _TREE_BUILDERS:
            raise ValueError("Unknown linkage type %s. "
                             "Valid options are %s" % (self.linkage,
                                                       _TREE_BUILDERS.keys()))
        tree_builder = _TREE_BUILDERS[self.linkage]

        connectivity = self.connectivity
        if self.connectivity is not None:
            if callable(self.connectivity):
                connectivity = self.connectivity(X)
            connectivity = check_array(
                connectivity, accept_sparse=['csr', 'coo', 'lil'])

        n_samples = len(X)
        compute_full_tree = self.compute_full_tree
        if self.connectivity is None:
            compute_full_tree = True
        if compute_full_tree == 'auto':
            if self.distance_threshold is not None:
                compute_full_tree = True
            else:
                # Early stopping is likely to give a speed up only for
                # a large number of clusters. The actual threshold
                # implemented here is heuristic
                compute_full_tree = self.n_clusters < max(100, .02 * n_samples)
        n_clusters = self.n_clusters
        if compute_full_tree:
            n_clusters = None

        # Construct the tree
        kwargs = {}
        if self.linkage != 'ward':
            kwargs['linkage'] = self.linkage
            kwargs['affinity'] = self.affinity

        distance_threshold = self.distance_threshold

        return_distance = distance_threshold is not None
        out = memory.cache(tree_builder)(X, connectivity,
                                         n_clusters=n_clusters,
                                         return_distance=return_distance,
                                         **kwargs)
        (self.children_,
         self.n_connected_components_,
         self.n_leaves_,
         parents) = out[:4]

        if distance_threshold is not None:
            distances = out[-1]
            self.distances_ = distances
            self.n_clusters_ = np.count_nonzero(
                distances >= distance_threshold) + 1
        else:
            self.n_clusters_ = self.n_clusters

        # Cut the tree
        if compute_full_tree:
            self.labels_ = _hc_cut(self.n_clusters_, self.children_,
                                   self.n_leaves_)
        else:
            labels = _hierarchical.hc_get_heads(parents, copy=False)
            # copy to avoid holding a reference on the original array
            labels = np.copy(labels[:n_samples])
            # Reassign cluster numbers
            self.labels_ = np.searchsorted(np.unique(labels), labels)
        return self