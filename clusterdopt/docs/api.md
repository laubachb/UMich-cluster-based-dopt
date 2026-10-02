# API reference

```python
from clusterdopt import ClusterDOpt, maxvol, gamma_rows, gamma_samples
```

The [user guide](user_guide.md) explains the method and how to use it.

---

## `ClusterDOpt`

```python
ClusterDOpt(n_clusters=6, rows_per_sample=1, n_solutions=10, n_neighbors_affinity=20, n_neighbors_assign=15,
            cluster_seed=42, maxvol_seed=300, tol=1.01, max_iters=300, n_jobs=None)
```

Cluster-local D-optimal extrapolation grade. The defaults reproduce the protocol of the manuscript.

### Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `n_clusters` | 6 | number of spectral clusters of the training samples |
| `rows_per_sample` | 1 | consecutive rows forming one sample; 3 for per-atom force components (x, y, z) |
| `n_solutions` | 10 | independent MaxVol solutions per cluster; γ is their mean |
| `n_neighbors_affinity` | 20 | neighbours per sample in the spectral-clustering graph |
| `n_neighbors_assign` | 15 | training samples whose distance-weighted vote assigns a new sample to a cluster |
| `cluster_seed` | 42 | random state of the spectral clustering (its k-means step) |
| `maxvol_seed` | 300 | solution t uses `numpy.random.default_rng(maxvol_seed + t)` |
| `tol` | 1.01 | MaxVol stops when no coefficient exceeds `tol` in absolute value |
| `max_iters` | 300 | maximum number of MaxVol row swaps per basis |
| `n_jobs` | None | parallel jobs for the nearest-neighbour searches (None = 1, −1 = all cores) |

### Attributes (set by `fit`)

| Attribute | Shape | Meaning |
|---|---|---|
| `n_columns_` | int | columns of the training matrix = rows of every basis |
| `n_clusters_` | int | clusters actually used (≤ `n_clusters` if clusters were merged) |
| `labels_` | (n_train_samples,) | cluster of every training sample, 0 … `n_clusters_` − 1 |
| `pivots_` | (n_solutions, n_clusters_, n_columns_) | training-matrix rows of every basis |
| `inverses_` | (n_solutions, n_clusters_, n_columns_, n_columns_) | inverse of every basis |
| `scaler_` | | `StandardScaler` fit on the training features |
| `features_` | (n_train_samples, rows_per_sample × n_columns_) | standardized training features |
| `assigner_` | | `KNeighborsClassifier` that assigns new samples to clusters |

### Methods

#### `fit(A) -> self`

Clusters the training matrix `A` (n_rows × n_columns) and selects the MaxVol bases of every cluster. `n_rows` must
be a multiple of `rows_per_sample`, and `A` must have full column rank.

Raises `ValueError` if `A` is not 2-D, its row count is not a multiple of `rows_per_sample`, or it is rank
deficient. Warns (`UserWarning`) when a cluster is merged into its nearest neighbour because it cannot support a
square basis.

#### `gamma(A, return_solutions=False, return_clusters=False)`

γ_cluster of every sample of `A` (n_rows × `n_columns_`), averaged over the MaxVol solutions.

Returns:
- `gamma`, shape (n_samples,);
- `per_solution`, shape (n_solutions, n_samples), if `return_solutions=True`;
- `clusters`, shape (n_samples,), if `return_clusters=True`.

With more than one output, a tuple is returned in this order.

#### `predict_cluster(A) -> ndarray`

Cluster of every sample of `A`, by the distance-weighted vote of its `n_neighbors_assign` nearest training samples
in the standardized feature space.

#### `save(path)` / `ClusterDOpt.load(path)`

Write the fitted model to a compressed `.npz` file of plain arrays (no pickle), and read it back. A loaded model
returns exactly the same `gamma` and `predict_cluster` results as the original.

---

## Low-level functions

These are used by `ClusterDOpt` and are useful on their own, for example to compute the conventional bulk grade
for comparison.

#### `maxvol(A, rng, tol=1.01, max_iters=300) -> (pivots, inverse)`

Rectangular MaxVol on the reference rows `A` (n_rows ≥ n_columns). It returns the indices of the `n_columns` rows
it selects and the inverse of the square submatrix `A[pivots]`.

- `rng` is a `numpy.random.Generator`. It draws the random initial rows, redrawing up to 30 times if they are
  singular, and falls back to pivoted QR after that.
- Rows are swapped while any coefficient of `A @ inverse` exceeds `tol`, up to `max_iters` swaps.
- Given the same generator state, it selects the same rows as the implementation used for the manuscript.

Raises `ValueError` if there are fewer rows than columns, and `numpy.linalg.LinAlgError` if `A` has rank < n_columns.

#### `gamma_rows(A, inverse) -> ndarray`

Per-row grade max_j |(**a** · inverse)_j|, shape (n_rows,). It is computed in chunks, so large matrices do not need
a full intermediate product in memory.

#### `gamma_samples(A, inverse, rows_per_sample=1) -> ndarray`

Per-sample grade: the maximum of `gamma_rows` over each sample's consecutive rows, shape (n_rows / rows_per_sample,).

---

## Command line

```
clusterdopt fit TRAIN.npy MODEL.npz [--n-clusters 6] [--rows-per-sample 1] [--n-solutions 10] [--n-jobs N]
clusterdopt gamma MODEL.npz NEW.npy GAMMA.npy [--clusters CLUSTERS.npy]
```

- `fit` reads a training matrix, fits a `ClusterDOpt` and saves it.
- `gamma` loads a saved model, grades a matrix and writes one γ_cluster per sample (and, with `--clusters`, the
  assigned clusters).
- `python3 -m clusterdopt.cli` is equivalent to the `clusterdopt` command.
