# clusterdopt user guide

This guide covers what `clusterdopt` computes, how to prepare its input, and how to use the result in active
learning. For a list of every function and parameter, see the [API reference](api.md).

- [1. Background](#1-background)
- [2. Preparing the input](#2-preparing-the-input)
- [3. Fitting a model](#3-fitting-a-model)
- [4. Grading new configurations](#4-grading-new-configurations)
- [5. Selecting frames for labeling](#5-selecting-frames-for-labeling)
- [6. Interpreting the grade](#6-interpreting-the-grade)
- [7. Choosing the number of clusters](#7-choosing-the-number-of-clusters)
- [8. Saving, loading and the command line](#8-saving-loading-and-the-command-line)
- [9. Reproducibility and performance](#9-reproducibility-and-performance)
- [10. Troubleshooting](#10-troubleshooting)

---

## 1. Background

### The D-optimal extrapolation grade

For a model that is linear in its parameters, such as ChIMES, a moment tensor potential (MTP) or SNAP, a prediction
for one training target is a dot product: F = **a** · **c**, where **c** holds the fitted coefficients and **a**
is a row of the design matrix **A**. Every force component, energy or stress in the training set contributes one
row.

D-optimal active learning (Podryabinkin & Shapeev, 2017) selects a square submatrix **Â** of the training rows with
the largest possible |det|, using the MaxVol algorithm. **Â** has one row per model parameter. A new row **a** is
then graded by

  γ(**a**) = max_j |(**a Â**⁻¹)_j| .

The vector **a Â**⁻¹ expresses the new row in terms of the basis rows. If every coefficient has magnitude at most 1,
the row lies inside the volume spanned by the basis (interpolation). If one coefficient is larger than 1, the row
reaches outside that volume (extrapolation), and labeling it would enlarge the basis. γ > 1 is the conventional
signal to send a configuration to DFT.

### Why one basis is not enough

The conventional grade γ_bulk selects one basis from the whole training set. Suppose the training set mixes
physically distinct regimes of different size and spread, for example cold molecular N₂ and hot, dissociated
nitrogen. MaxVol maximizes volume, so the basis rows come mostly from the regime whose descriptors spread furthest.
In the ChIMES nitrogen data, the molecular clusters hold 25% of the training rows but supply about 3 of the 42 basis
rows.

A new configuration in the under-represented regime is then expressed mainly through rows of other regimes. It can
be far from every training configuration of its own regime and still get γ_bulk < 1. In the nitrogen study, γ_bulk
had no ranking ability for the force error inside the molecular clusters (Spearman ≈ −0.05).

### The cluster-local grade

`clusterdopt` computes γ_cluster instead:

1. **Cluster** the training samples by their descriptor rows (spectral clustering, no labels needed).
2. **Select one MaxVol basis per cluster** from that cluster's rows only.
3. **Assign** a new sample to a cluster (by its nearest training samples), and **grade** it against that cluster's
   basis.

Each regime now has a basis that resolves its own variation. In the nitrogen study, γ_cluster restored the ranking
inside the molecular clusters (Spearman 0.21–0.30). At a 10% labeling budget it recovered 43% of their highest-error
candidates, against 10% for γ_bulk.

---

## 2. Preparing the input

`clusterdopt` takes the **design matrix** of your linear model as a NumPy array of shape (rows, columns). The columns
are the model parameters, and the rows are training targets. Use the matrix that your fitting code builds, without
the target vector **b** and without per-row fitting weights.

### What a "sample" is: `rows_per_sample`

| Your rows are | Use | A sample is | Its γ is |
|---|---|---|---|
| independent training points (any linear model) | `rows_per_sample=1` (default) | one row | the row's γ |
| per-atom force components x, y, z, three consecutive rows per atom | `rows_per_sample=3` | one atom | the largest of its three row grades |

With `rows_per_sample=3`, the three rows of an atom are concatenated into one feature vector for clustering, so an
atom is always assigned to one cluster as a whole. This is the protocol of the manuscript. The number of rows must
be a multiple of `rows_per_sample`, and the rows of each sample must be consecutive.

### Requirements

- **Same columns** in the training matrix and in every matrix you grade, in the same order (same descriptor
  hyperparameters).
- **Full column rank.** Every basis is square (columns × columns), so the training matrix needs as many linearly
  independent rows as it has columns. All-zero columns, such as parameters of an interaction type that never occurs,
  make that impossible. `fit` checks this and stops with an explanation; remove those columns from every matrix:
  ```python
  keep = np.abs(A_train).max(axis=0) > 0
  A_train, A_new = A_train[:, keep], A_new[:, keep]
  ```
- **One kind of row.** Mixing force, energy and stress rows in one matrix mixes very different scales. The method
  was validated on force rows only. If your fit includes energy or stress rows, remove them before passing the
  matrix.

### ChIMES (chimes_lsq)

With `FITENER` and `FITSTRS` set to `false` in `fm_setup.in`, `chimes_lsq` writes a force-only design matrix to
`A.txt`. It has three rows (fx, fy, fz) per atom, with atoms in frame order and frames in the order of the trajectory
list. Run `chimes_lsq` once on the training frames and once on the candidate frames, with the same hyperparameters:

```python
import numpy as np
from clusterdopt import ClusterDOpt

A_train = np.loadtxt("train/A.txt")         # 3 rows per atom
A_cand = np.loadtxt("candidates/A.txt")
model = ClusterDOpt(rows_per_sample=3).fit(A_train)
gamma_atoms = model.gamma(A_cand)           # one value per candidate atom
```

[`examples/chimes_rank_frames.py`](../examples/chimes_rank_frames.py) does this end to end and ranks the candidate
frames (Section 5). `np.loadtxt` is slow for large files; convert to `.npy` once with `np.save` if you reuse a
matrix.

### Other linear models

Any matrix whose rows are linear-model targets works. For SNAP-type or MTP-type models, export the rows that your
fitting code assembles (for example, per-atom force rows of the descriptor derivatives). Use `rows_per_sample=3` if
they come as consecutive x, y, z rows per atom. Make sure the descriptors of new configurations are computed with
exactly the same settings as those of the training set.

---

## 3. Fitting a model

```python
model = ClusterDOpt(n_clusters=6, rows_per_sample=3).fit(A_train)
```

`fit` performs these steps:

1. **Features.** Each sample's rows are concatenated, and every feature is standardized (zero mean, unit variance)
   over the training samples.
2. **Clustering.** Spectral clustering of the standardized samples into `n_clusters` clusters. It uses a
   nearest-neighbour graph with `n_neighbors_affinity` = 20 neighbours, and k-means on the spectral embedding with
   random state `cluster_seed` = 42. Spectral clustering partitions the neighbour graph, so clusters need not be
   convex or of similar spread. This matters because a compact regime can sit inside a broad one.
3. **Merging.** A cluster that cannot support a square basis (fewer independent rows than columns) is merged into
   the cluster with the nearest centroid, with a warning (Section 10).
4. **Bases.** For each MaxVol solution t = 0 … `n_solutions` − 1, a random generator is seeded with
   `maxvol_seed + t`, and one basis is selected per cluster, in increasing cluster order. MaxVol starts from random
   rows and swaps in the row with the largest coefficient until none exceeds `tol` = 1.01.

After fitting, the model holds:

| Attribute | Meaning |
|---|---|
| `model.labels_` | cluster of every training sample |
| `model.n_clusters_` | number of clusters actually used (after any merging) |
| `model.pivots_` | training-matrix row indices of every basis, shape (n_solutions, n_clusters_, n_columns) |
| `model.inverses_` | inverse of every basis, shape (n_solutions, n_clusters_, n_columns, n_columns) |

Check the clusters before relying on them. `np.bincount(model.labels_)` gives the cluster sizes. If you know what each
training configuration is (temperature, phase, composition), cross-tabulate it against `labels_` to see whether the
clusters follow physical regimes.

### Why several MaxVol solutions?

MaxVol finds a locally maximal volume, and different starting rows give different bases. Individual solutions can
rank error quite differently: on the nitrogen candidates, the Spearman correlation of single-solution γ_bulk with
error ranged from −0.01 to 0.27. `clusterdopt` therefore averages γ over `n_solutions` = 10 solutions by default.
Using fewer solutions is faster but noisier.

---

## 4. Grading new configurations

```python
gamma = model.gamma(A_new)                                   # one value per sample
gamma, per_solution, clusters = model.gamma(A_new, return_solutions=True, return_clusters=True)
clusters = model.predict_cluster(A_new)                      # clusters only
```

- Every new sample is assigned to the cluster that wins a distance-weighted vote of its `n_neighbors_assign` = 15
  nearest training samples (in the standardized feature space of Section 3).
- It is graded against each of that cluster's bases, and the grades are averaged over solutions.
- `per_solution` has shape (n_solutions, n_samples). Use it to build your own flag rule, for example "γ > 1 in at
  least 80% of solutions", which the manuscript used for its MD flags:
  ```python
  robust_flag = (per_solution > 1).mean(axis=0) >= 0.8
  ```

New samples need descriptors only, no DFT labels. Every training sample has γ ≤ `tol` against the bases of its own
cluster, so values clearly above 1 mean the sample extends its cluster.

### Comparing with the conventional grade

To see what the cluster-local basis adds on your data, compute γ_bulk with the same MaxVol routine:

```python
from clusterdopt import maxvol, gamma_samples
import numpy as np

_, inv_bulk = maxvol(A_train, np.random.default_rng(0))
gamma_bulk = gamma_samples(A_new, inv_bulk, rows_per_sample=3)
```

---

## 5. Selecting frames for labeling

DFT labels whole configurations (frames), not single atoms. To score a frame, take the largest atom grade in it:

```python
gamma_atoms = model.gamma(A_cand)                        # one value per atom
frame_of_atom = np.repeat(np.arange(n_frames), atoms_per_frame)
frame_score = np.full(n_frames, -np.inf)
np.maximum.at(frame_score, frame_of_atom, gamma_atoms)

budget = 20
to_label = np.argsort(-frame_score)[:budget]             # highest-scoring frames first
```

A typical active-learning iteration:

1. Fit `ClusterDOpt` on the current training matrix.
2. Grade the candidate configurations, for example frames from MD with the current model.
3. Label the top-ranked frames with DFT, add them to the training set and refit the potential.
4. Refit `ClusterDOpt` on the enlarged training matrix before the next iteration, because the clusters and bases
   depend on the training set.

[`examples/chimes_rank_frames.py`](../examples/chimes_rank_frames.py) implements steps 1–2 for ChIMES and writes a
CSV of frames ranked by their highest atom grade.

---

## 6. Interpreting the grade

These observations come from the nitrogen study of the manuscript. Use them as guidance for your own system, not as
guarantees.

**Ranking is more reliable than the threshold.** γ_cluster > 1 means "outside the volume of this cluster's training
basis". In regimes the model has never been trained to resolve, a large share of atoms can exceed 1. In MD of the
DFT-only nitrogen model at 300 K and 2000 K, 51–56% of atom-frames had γ_cluster > 1, and the flagged atoms were
not concentrated among the highest-error atoms. Ranking frames and labeling a fixed budget from the top was the more useful strategy.

**Where γ_cluster helps most.** The gain is largest in clusters that the bulk basis under-represents, typically small
or narrow regimes. There, γ_bulk flags almost nothing and does not rank error, while γ_cluster flags precisely
(72–79% of its flags were in the cluster's top-10% error) and ranks error. In clusters that dominate the bulk basis,
the two grades ranked error about equally, and γ_bulk's threshold flags were more precise (86–91% vs 35–42%). If
your training set has no under-represented regime, expect little difference from the conventional grade.

**Scale differs between clusters.** Each cluster has its own basis, so γ_cluster values from different clusters
are not on exactly the same scale. Pooled rankings across clusters are dominated by which regime a sample belongs
to. Compare grades within a cluster when you need a fine ranking. `return_clusters=True` gives you the cluster of
each sample.

**Effect on the refit.** In the nitrogen study, frames chosen by γ_cluster lowered the error in the neglected regime
2–3× faster than random frames, and the refit models were more stable in MD. With a small model (42 parameters),
adding data from one regime temporarily raised the error in others until frames from those regimes were also
selected. Monitor the error in every regime, not only the one being improved.

**No early warning.** In the nitrogen MD, neither grade rose before an MD instability (collapse) happened. The
stability benefit came from the data γ_cluster selected for the refit, not from detecting instabilities as they
developed.

---

## 7. Choosing the number of clusters

`n_clusters` is the only parameter you are likely to change. The default of 6 is the value used in the manuscript.

- **Too few clusters** merge distinct regimes, and the small regime again shares a basis with a large one. In the
  nitrogen sweep (k = 4–8, three clustering seeds each), γ_cluster beat γ_bulk for every k. Its benefit was weaker at
  k = 4–5 and levelled off from k = 6.
- **More clusters** split regimes further. This is harmless as long as every cluster keeps many more rows than there
  are columns. The smallest nitrogen cluster had 850 atoms (2,550 rows) for 42 columns.
- **The clustering seed** hardly mattered: at k = 6, the three seeds gave nearly the same partition (adjusted Rand
  index 0.99).

A practical approach:
1. Start with the default.
2. Look at the cluster sizes and, if you have metadata, at the cluster composition.
3. Increase `n_clusters` if a physically distinct regime shares a cluster with a much larger one.
4. Decrease it if clusters come out with only a few times more rows than columns.

---

## 8. Saving, loading and the command line

```python
model.save("model.npz")
model = ClusterDOpt.load("model.npz")
```

The file holds plain NumPy arrays (no pickle): the parameters, the scaler, the standardized training features (for
assigning new samples), the training labels and every basis inverse. Its size is dominated by the training features
(samples × rows_per_sample × columns × 8 bytes).

The command-line tool works on `.npy` files:

```bash
clusterdopt fit train.npy model.npz --n-clusters 6 --rows-per-sample 3 --n-solutions 10
clusterdopt gamma model.npz new.npy gamma.npy --clusters clusters.npy
```

`gamma.npy` holds one γ_cluster per sample, and `clusters.npy` the assigned clusters. If the `clusterdopt` command
is not on your `PATH` after installation, use `python3 -m clusterdopt.cli` instead.

---

## 9. Reproducibility and performance

**Reproducibility.** With the same input, parameters and library versions, results are deterministic. The
clustering is seeded by `cluster_seed`, and MaxVol solution t by `maxvol_seed + t`. Spectral clustering can change
between scikit-learn versions. The manuscript used scikit-learn 1.8.0 and NumPy 2.4.1.

The MaxVol routine selects the same rows as the implementation used for the manuscript, given the same random
generator state. It is faster because it updates the coefficient matrix after each row swap instead of recomputing
it. The manuscript drew the bulk basis from the same generator before the cluster bases, so the default solutions
here are different but statistically equivalent draws. [`examples/nitrogen_check.py`](../examples/nitrogen_check.py)
reproduces the manuscript's γ_cluster exactly by drawing in the manuscript's order.

**Performance.** These timings are for the nitrogen data on one core: 10,560 training atoms × 126 features, 42
columns, 10 solutions.

| Step | Time |
|---|---|
| `fit` (mostly spectral clustering) | 46 s |
| `gamma` on 47,048 atoms | 4 s |

Spectral clustering dominates the cost, and its time and memory grow with the number of training samples. The
package has not been tested beyond about 10⁴ training samples. Grading is a matrix product and scales linearly with
the number of new rows. Set `n_jobs=-1` to parallelize the nearest-neighbour searches.

---

## 10. Troubleshooting

**`ValueError: the training matrix has rank r < n columns`.** Some columns are all zero or linearly dependent. Remove
them from the training matrix and from every matrix you grade (Section 2).

**`UserWarning: cluster c (...) cannot support a rank-n basis; merged into cluster d`.** A cluster had fewer
linearly independent rows than there are columns, so it was merged with its nearest neighbour.
`model.n_clusters_` reports how many clusters remain. If this happens often, lower `n_clusters`.

**`UserWarning: Graph is not fully connected, spectral embedding may not work as expected`.** This comes from
scikit-learn when some groups of training samples have no neighbours in common, which happens when regimes are
completely separated. The clustering usually still follows those groups. Check `np.bincount(model.labels_)`, and
increase `n_neighbors_affinity` if the clusters look wrong.

**`ValueError: ... rows is not a multiple of rows_per_sample`.** The matrix does not consist of whole samples.
With ChIMES, check that the design matrix is force-only (Section 2).

**`ValueError: expected n columns, got m`.** The matrix you are grading was built with different descriptor
settings than the training matrix.

**γ_cluster is large for almost everything.** The new configurations are far from the training data of their
clusters, as in MD of a model that has just left its training coverage. Rank and label a fixed budget rather than
thresholding at 1 (Section 6).
