"""Cluster-local D-optimality: cluster the training rows, select a MaxVol basis per cluster, grade new rows against
the basis of their own cluster (gamma_cluster).

Protocol (as in the manuscript):
  features    each sample's rows concatenated (rows_per_sample x n_columns values), standardized on the training set
  clustering  SpectralClustering(n_clusters, nearest-neighbour affinity with 20 neighbours, k-means label
              assignment, random_state=42) on the training samples
  assignment  a new sample takes the distance-weighted vote of its 15 nearest training samples
  bases       for MaxVol solution t, rng = default_rng(maxvol_seed + t); one basis per cluster, clusters in
              increasing label order, from the rows of that cluster's training samples
  gamma       per row max_j |(a A_hat^-1)_j|; per sample the maximum over its rows; averaged over solutions

A sample is one row by default. For per-atom force-component rows (x, y, z), use rows_per_sample=3: the three rows
are clustered together and the atom's grade is the maximum over them.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from .maxvol import gamma_samples, maxvol

__version__ = "0.1.0"


class ClusterDOpt:
    """Cluster-local D-optimal extrapolation grade.

    Parameters
    ----------
    n_clusters : int, default 6
        Number of spectral clusters of the training samples.
    rows_per_sample : int, default 1
        Number of consecutive rows that form one sample. 1: every row is a training point. 3: per-atom force
        components (x, y, z) of one atom, clustered together and graded by the largest of the three row grades.
    n_solutions : int, default 10
        Independent random-restart MaxVol solutions per cluster. The returned gamma is their mean.
    n_neighbors_affinity : int, default 20
        Neighbours per sample in the spectral-clustering affinity graph.
    n_neighbors_assign : int, default 15
        Training samples whose distance-weighted vote assigns a new sample to a cluster.
    cluster_seed : int, default 42
        Random state of the spectral clustering.
    maxvol_seed : int, default 300
        MaxVol solution t uses numpy.random.default_rng(maxvol_seed + t).
    tol : float, default 1.01
        MaxVol stops when no row has a coefficient larger than tol in absolute value.
    max_iters : int, default 300
        Maximum number of MaxVol row swaps per basis.
    n_jobs : int or None
        Parallel jobs for the nearest-neighbour searches (None = 1, -1 = all cores).

    Attributes (after fit)
    ----------------------
    n_columns_ : number of columns of the training matrix (= rows of every basis).
    n_clusters_ : number of clusters actually used (smaller than n_clusters if clusters were merged).
    labels_ : (n_training_samples,) cluster of every training sample.
    pivots_ : (n_solutions, n_clusters_, n_columns_) training-matrix rows of every basis.
    inverses_ : (n_solutions, n_clusters_, n_columns_, n_columns_) inverse of every basis.

    Examples
    --------
    >>> model = ClusterDOpt(n_clusters=6, rows_per_sample=3).fit(A_train)     # doctest: +SKIP
    >>> gamma = model.gamma(A_new)                                           # doctest: +SKIP
    >>> flagged = gamma > 1                                                  # doctest: +SKIP
    """

    def __init__(self, n_clusters=6, rows_per_sample=1, n_solutions=10, n_neighbors_affinity=20,
                 n_neighbors_assign=15, cluster_seed=42, maxvol_seed=300, tol=1.01, max_iters=300, n_jobs=None):
        self.n_clusters = n_clusters
        self.rows_per_sample = rows_per_sample
        self.n_solutions = n_solutions
        self.n_neighbors_affinity = n_neighbors_affinity
        self.n_neighbors_assign = n_neighbors_assign
        self.cluster_seed = cluster_seed
        self.maxvol_seed = maxvol_seed
        self.tol = tol
        self.max_iters = max_iters
        self.n_jobs = n_jobs

    # ------------------------------------------------------------------ helpers
    def _params(self):
        return {k: getattr(self, k) for k in ("n_clusters", "rows_per_sample", "n_solutions", "n_neighbors_affinity",
                                               "n_neighbors_assign", "cluster_seed", "maxvol_seed", "tol",
                                               "max_iters", "n_jobs")}

    def _check(self, A, n_columns=None):
        A = np.asarray(A, dtype=np.float64)
        if A.ndim != 2:
            raise ValueError(f"expected a 2-D matrix (rows x columns), got shape {A.shape}")
        if A.shape[0] % self.rows_per_sample:
            raise ValueError(f"{A.shape[0]} rows is not a multiple of rows_per_sample={self.rows_per_sample}")
        if n_columns is not None and A.shape[1] != n_columns:
            raise ValueError(f"expected {n_columns} columns, got {A.shape[1]}")
        return A

    def _features(self, A):
        return A.reshape(A.shape[0] // self.rows_per_sample, -1)

    def _sample_rows(self, samples):
        rps = self.rows_per_sample
        return (np.asarray(samples)[:, None] * rps + np.arange(rps)[None, :]).ravel()

    def _merge_small_clusters(self, A, Z, labels):
        """merge clusters that cannot support a full-rank basis into the cluster with the nearest centroid."""
        r = A.shape[1]
        while True:
            ids = np.unique(labels)
            bad = [c for c in ids
                   if np.linalg.matrix_rank(A[self._sample_rows(np.where(labels == c)[0])]) < r]
            if not bad:
                break
            if len(ids) == 1:
                raise np.linalg.LinAlgError(f"the training rows have rank < {r}: no {r} x {r} basis exists")
            c = min(bad, key=lambda c: (labels == c).sum())
            cent = {k: Z[labels == k].mean(axis=0) for k in ids}
            target = min((k for k in ids if k != c), key=lambda k: np.linalg.norm(cent[k] - cent[c]))
            warnings.warn(f"cluster {c} ({(labels == c).sum()} samples) cannot support a rank-{r} basis; "
                          f"merged into cluster {target}", stacklevel=3)
            labels = np.where(labels == c, target, labels)
        return np.unique(labels, return_inverse=True)[1]

    # ------------------------------------------------------------------ fitting
    def fit(self, A):
        """Cluster the training rows and select the MaxVol bases of every cluster.

        A : (n_rows, n_columns) training design matrix, n_rows a multiple of rows_per_sample. Needs full column rank.
        Returns self.
        """
        A = self._check(A)
        rank = np.linalg.matrix_rank(A)
        if rank < A.shape[1]:
            raise ValueError(
                f"the training matrix has rank {rank} < {A.shape[1]} columns, so no square MaxVol basis exists. "
                "Remove all-zero or linearly dependent columns (e.g. keep = np.abs(A).max(axis=0) > 0; A = A[:, keep]) "
                "and apply the same column selection to the matrices you grade.")
        self.n_columns_ = A.shape[1]
        X = self._features(A)
        self.scaler_ = StandardScaler().fit(X)
        Z = self.scaler_.transform(X)
        labels = SpectralClustering(n_clusters=self.n_clusters, affinity="nearest_neighbors",
                                    n_neighbors=self.n_neighbors_affinity, assign_labels="kmeans",
                                    random_state=self.cluster_seed, n_jobs=self.n_jobs).fit_predict(Z)
        self.labels_ = self._merge_small_clusters(A, Z, labels)
        self.n_clusters_ = int(self.labels_.max()) + 1
        self._fit_assigner(Z)
        r = self.n_columns_
        self.pivots_ = np.empty((self.n_solutions, self.n_clusters_, r), dtype=np.int64)
        self.inverses_ = np.empty((self.n_solutions, self.n_clusters_, r, r))
        rows = [self._sample_rows(np.where(self.labels_ == c)[0]) for c in range(self.n_clusters_)]
        for t in range(self.n_solutions):
            rng = np.random.default_rng(self.maxvol_seed + t)
            for c in range(self.n_clusters_):
                piv, inv = maxvol(A[rows[c]], rng, tol=self.tol, max_iters=self.max_iters)
                self.pivots_[t, c] = rows[c][piv]          # rows of the training matrix
                self.inverses_[t, c] = inv
        return self

    def _fit_assigner(self, Z):
        self.features_ = Z
        self.assigner_ = KNeighborsClassifier(n_neighbors=self.n_neighbors_assign, weights="distance",
                                              n_jobs=self.n_jobs).fit(Z, self.labels_)

    # ------------------------------------------------------------------ prediction
    def predict_cluster(self, A):
        """Cluster of every sample of A (n_rows x n_columns_), by the vote of the nearest training samples."""
        A = self._check(A, self.n_columns_)
        return self.assigner_.predict(self.scaler_.transform(self._features(A)))

    def gamma(self, A, return_solutions=False, return_clusters=False):
        """gamma_cluster of every sample of A (n_rows x n_columns_), averaged over the MaxVol solutions.

        Each sample is assigned to a cluster (predict_cluster) and graded against that cluster's bases:
        gamma = max_j |(a A_hat^-1)_j|, maximized over the sample's rows. gamma <= 1 (up to tol) inside the volume
        spanned by the cluster's basis, > 1 outside it (extrapolation within the cluster).

        return_solutions : also return the per-solution grades, shape (n_solutions, n_samples).
        return_clusters : also return the cluster of every sample.
        """
        A = self._check(A, self.n_columns_)
        labels = self.predict_cluster(A)
        G = np.empty((self.n_solutions, labels.size))
        for c in np.unique(labels):
            samples = np.where(labels == c)[0]
            Ac = A[self._sample_rows(samples)]
            for t in range(self.n_solutions):
                G[t, samples] = gamma_samples(Ac, self.inverses_[t, c], self.rows_per_sample)
        out = (G.mean(axis=0),)
        if return_solutions:
            out += (G,)
        if return_clusters:
            out += (labels,)
        return out if len(out) > 1 else out[0]

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        """Write the fitted model to an .npz file (plain arrays, no pickle). Restore it with ClusterDOpt.load(path)."""
        np.savez_compressed(path, params=json.dumps(self._params()), version=__version__,
                            n_columns=self.n_columns_, scaler_mean=self.scaler_.mean_,
                            scaler_scale=self.scaler_.scale_, features=self.features_,
                            labels=self.labels_, pivots=self.pivots_, inverses=self.inverses_)

    @classmethod
    def load(cls, path):
        """Read a model written by save()."""
        d = np.load(path, allow_pickle=False)
        model = cls(**json.loads(str(d["params"])))
        model.n_columns_ = int(d["n_columns"])
        model.scaler_ = StandardScaler()
        model.scaler_.mean_, model.scaler_.scale_ = d["scaler_mean"], d["scaler_scale"]
        model.scaler_.var_ = model.scaler_.scale_ ** 2
        model.scaler_.n_features_in_ = model.scaler_.mean_.size
        model.labels_ = d["labels"]
        model.n_clusters_ = int(model.labels_.max()) + 1
        model.pivots_, model.inverses_ = d["pivots"], d["inverses"]
        model._fit_assigner(d["features"])
        return model
