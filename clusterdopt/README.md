# clusterdopt

**Cluster-local D-optimal extrapolation grades for linear machine-learned interatomic potentials** (ChIMES, MTP,
SNAP, or any model that is linear in its parameters).

D-optimal active learning flags a new configuration for DFT when its extrapolation grade γ, computed against a
MaxVol basis of training rows, exceeds 1. The usual grade uses **one** basis from the whole training set. When the
training set mixes regimes of different size, that basis comes almost entirely from the large regime. New
configurations of a small regime then get γ < 1 even when they are far from all of its training data.

`clusterdopt` clusters the training set, selects **one basis per cluster**, and grades each new configuration
against the basis of its own cluster (γ_cluster):

```
training design matrix ──► spectral clustering ──► MaxVol basis per cluster
                                                          │
new configuration ──► assign to cluster (nearest neighbours) ──► γ_cluster against that cluster's basis
```

On the ChIMES nitrogen data of the accompanying paper, γ_cluster recovered 43% of the highest-error candidates of
the under-represented molecular regime at a 10% labeling budget, against 10% for the conventional grade.

## Install

```bash
git clone https://github.com/laubachb/UMich-cluster-based-dopt.git && cd UMich-cluster-based-dopt
pip install -e clusterdopt               # requires Python ≥ 3.9, numpy, scipy, scikit-learn
```

To install straight from GitHub: `pip install "git+https://github.com/laubachb/UMich-cluster-based-dopt.git#subdirectory=clusterdopt"`.

## Quickstart

```python
import numpy as np
from clusterdopt import ClusterDOpt

A_train = np.load("A_train.npy")         # training design matrix: rows = training targets, columns = parameters
A_new = np.load("A_new.npy")             # rows of new configurations, same columns

model = ClusterDOpt().fit(A_train)       # 6 spectral clusters, one MaxVol basis per cluster
gamma = model.gamma(A_new)               # one γ_cluster per row; > 1 = extrapolation within its cluster
```

**Per-atom force rows** (x, y, z as three consecutive rows per atom, as written by ChIMES `chimes_lsq`): pass
`rows_per_sample=3`. Each atom is then clustered as a unit and graded by the largest of its three row grades:

```python
model = ClusterDOpt(rows_per_sample=3).fit(A_train)
gamma_atoms = model.gamma(A_new)         # one value per atom
```

**Ranking frames for DFT:** score each frame by its largest atom grade and label the top of the list
(see [user guide §5](docs/user_guide.md#5-selecting-frames-for-labeling)).

**Save and reuse:** `model.save("model.npz")`, then `ClusterDOpt.load("model.npz")`.

**Command line:**
```bash
clusterdopt fit A_train.npy model.npz --rows-per-sample 3
clusterdopt gamma model.npz A_new.npy gamma.npy --clusters clusters.npy
```

## Documentation

| | |
|---|---|
| [User guide](docs/user_guide.md) | background, preparing the input (incl. ChIMES), fitting, grading, frame selection, how to interpret γ_cluster, choosing the number of clusters, troubleshooting |
| [API reference](docs/api.md) | every class, function, parameter and attribute |
| [`examples/quickstart.py`](examples/quickstart.py) | runnable demo on synthetic data (no external files) |
| [`examples/chimes_rank_frames.py`](examples/chimes_rank_frames.py) | rank candidate frames for DFT from chimes_lsq `A.txt` files |
| [`examples/nitrogen_check.py`](examples/nitrogen_check.py) | reproduces the paper's clusters and γ_cluster (needs the paper pipeline's `retrospective/work/`) |

## Key defaults

| Parameter | Default | When to change it |
|---|---|---|
| `n_clusters` | 6 | if a distinct regime shares a cluster with a much larger one (raise it), or clusters come out tiny (lower it); see [user guide §7](docs/user_guide.md#7-choosing-the-number-of-clusters) |
| `rows_per_sample` | 1 | 3 for per-atom force-component rows |
| `n_solutions` | 10 | fewer for speed; γ is averaged over solutions because MaxVol solutions are not unique |

Every other parameter keeps the paper's protocol. See the [API reference](docs/api.md).

## Good to know

- **Ranking beats thresholding** when new configurations are far from the training data: rank and label a fixed
  budget rather than labeling everything with γ > 1 ([user guide §6](docs/user_guide.md#6-interpreting-the-grade)).
- **The gain is in under-represented regimes.** Where a regime already dominates the bulk basis, γ_cluster and the
  conventional grade rank error about equally.
- **The training matrix needs full column rank.** Remove all-zero columns from all matrices before fitting.
- **Refit `ClusterDOpt` after every active-learning iteration**, because the clusters and bases depend on the
  training set.

## Relation to the paper

This package implements the method of Laubach & Lindsey, *Cluster-Local D-Optimality Recovers Extrapolation Signals
Lost in Under-Represented Regimes of Machine-Learned Interatomic Potentials* (in preparation). The rest of this
repository reproduces the paper's results and does not depend on the package.

`examples/nitrogen_check.py` validates the package against the paper:
- the training and candidate clusters are identical;
- with random draws in the paper's order, γ_cluster is identical for all 10 MaxVol solutions;
- with the package's own draws, the agreement is statistical: 1.27% vs 1.24% of candidates have γ > 1, and the
  recall of the highest-error molecular candidates at a 10% budget is 0.44 vs 0.43.

The MaxVol routine selects the same rows as the paper's implementation from the same random state. It is faster
because it updates the coefficient matrix after each row swap instead of recomputing it.

## Tests

```bash
pip install -e "clusterdopt[test]"
python3 -m pytest clusterdopt/tests
```
