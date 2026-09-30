# Retrospective study: bulk vs cluster-local extrapolation grades on the ChIMES nitrogen candidates (Figs. 1–5)

## Question

Active learning of linear MLIPs (ChIMES, MTP, SNAP) usually flags configurations for DFT with a D-optimal
extrapolation grade γ. γ is computed against **one** MaxVol reference basis selected from the whole training set
(here called γ_bulk). The nitrogen training set mixes a small molecular regime (cold N₂) with a large, partly
dissociated high-temperature regime. A basis chosen to maximize volume over the whole set sits mostly in the large,
varied regime. This study tests whether grading each atom against a MaxVol basis built only from its own cluster
(γ_cluster) recovers the variation that γ_bulk loses. The candidate pools of the original active-learning campaign
have DFT forces, so the grades can be compared with a model's true force error without new calculations.

## Pipeline

Run from the repository root. Intermediates go to `retrospective/work/` (git-ignored); the small tables behind the
figures go to `retrospective/results/` (tracked); figures go to `figures/`.

| Step | Script | What it does | Needs | Time |
|---|---|---|---|---|
| 1 | `01_descriptors.py` | chimes_lsq force-only design matrix for all 646 frames (`fm_setup.in`): 3 rows per atom × 42 columns → `A_atomic.npy`, frame and atom manifests | `CHIMES_LSQ`, `MPIRUN` | minutes on 48 cores |
| 2 | `02_cluster.py` | spectral clustering of DFT atoms, 15-NN assignment of candidate atoms, UMAP embedding | – | ~5 min |
| 3 | `03_errors.py` | 75/25 DFT frame split (seed 42) → `results/dft_holdout.json`; ridge fit on the training frames; per-atom candidate force error → `work/atom_errors.csv` | – | <1 min |
| 4 | `04_gamma_ensemble.py` | γ_bulk and γ_cluster for all candidates, 10 MaxVol solutions; flag counts; gradient-boosting error regressors (GroupKFold and temporal splits) → `results/{outliers,aggregate,by_cluster,recall}.csv` | – | ~1 h |
| 5 | `05_controls.py 5` | control arms (raw grades, label only, γ_bulk + label, …), 5 MaxVol solutions → `results/controls.csv` | – | ~30 min |
| 6 | `06_figures.py` | Figs. 1–5 → `figures/`; cluster × state-point table → `results/cluster_composition.csv` | – | <1 min |

`slurm/run_retrospective.cmd` runs all six steps on one node (TACC Stampede3 example).

## Methods

**Descriptors.** ChIMES 2+3-body Chebyshev basis: orders 12/5, cutoffs 8.0/5.0 Å, inner cutoff 0.86 Å, Morse
λ = 1.09 Å, Tersoff smooth cutoff (f_O = 0.75), giving 42 coefficients. Forces are linear in the coefficients,
F_iα = a_iα · c. The rows a_iα (one per atom and Cartesian component) are the descriptors used for clustering, for γ
and for fitting.

**Clustering.** Each atom's three rows are concatenated (126 values) and standardized on the DFT atoms. Then:
- SpectralClustering(k = 6, 20-nearest-neighbour affinity, k-means label assignment, random_state = 42) on the
  10,560 DFT atoms.
- Each candidate atom takes the distance-weighted vote of its 15 nearest DFT atoms.

Clusters c0, c3 and c4 (25.2% of atoms) contain the 300 K and 2000 K molecular configurations and are called the
minority (or "hidden") clusters. c1, c2 and c5 contain the 5000–8000 K configurations.

**Error reference.** 90 of the 120 DFT-MD frames (15 of 20 per state point) are fit by ridge least squares on
forces (λ = 1e-8). The per-atom error of every candidate is the mean absolute force-component deviation
(Hartree/bohr). This model stands in for the preliminary model at the start of an active-learning campaign.

**Extrapolation grades** (`common/dopt.py`). Rectangular MaxVol (tolerance 1.01) selects a 42 × 42 submatrix  of
locally maximal |det| from the reference rows. γ(a) = max_j |(a Â⁻¹)_j|, and the per-atom γ is the maximum over the
three component rows.
- γ_bulk uses all DFT atoms as the reference.
- γ_cluster uses only the DFT atoms of the candidate's cluster.

MaxVol solutions from different random initial pivots differ, so every number is a mean over independent
solutions (seeds 300 + t).

**Evaluation.** Rankings are scored with:
- Spearman correlation with the error;
- AUROC for the 10% of atoms with the highest error;
- recall of those atoms at a labelling budget.

Scores are raw grades or out-of-fold predictions of `HistGradientBoostingRegressor`. The regressors use 5-fold
GroupKFold by frame, or train on the alc2 pool and test on alc3 or alc4.

## Results and how to read them

| Figure | Content | Headline |
|---|---|---|
| 1 | UMAP of DFT atoms by cluster and state point | Clustering separates molecular (c0, c3, c4) from dissociated (c1, c2, c5) nitrogen without temperature labels |
| 2 | Candidates with γ > 1 per cluster | Minority clusters: γ_bulk flags 3–5 atoms each, γ_cluster flags 72–101 that γ_bulk misses. Majority cluster c1: γ_bulk flags more (193) |
| 3 | Pooled ranking, γ_bulk regressor vs γ_cluster + cluster-label regressor | Spearman 0.60–0.68 vs 0.11–0.13. **Most of this gap comes from the cluster label** (label alone 0.62, γ_bulk + label 0.65; `results/controls.csv`) |
| 4 | Within-cluster ranking margin | Minority clusters +0.17 to +0.21 Spearman; c1 0.00 ± 0.05. Raw grades give the same picture: γ_bulk −0.05 to −0.06, γ_cluster 0.21–0.30 within the minority clusters |
| 5 | Recall of top-10% error minority candidates vs budget | At 10%: 0.42 (γ_cluster) vs 0.11 (γ_bulk). Raw γ_cluster alone gives 0.42 and the label alone 0.10, so the gain comes from the cluster-local grade |

Supported claim: γ_bulk does not rank error within under-represented regimes, and γ_cluster does. Pooled gains
mainly reflect the regime label. Figs. 3–5 in the current draft plot regressor scores. The manuscript notes planned
revisions to add the control arms (Fig. 3) and to plot raw grades (Figs. 4–5); the numbers for those revisions are
already in `results/controls.csv`.

## Notes

- **MaxVol reimplementation.** The original γ code used in the ChIMES workflow was unavailable, and `common/dopt.py`
  reimplements rectangular MaxVol. Its γ_bulk agrees with the saved output of the original (Pearson 0.989; 0.82% vs
  0.85% of candidates with γ > 1). Across solutions, Spearman(γ_bulk, error) ranges from −0.01 to 0.27, which is why
  all results are averaged.
- The earlier Euler-characteristic clustering variant, the selection → refit experiments ("Mode I") and other
  exploratory analyses are not part of this pipeline and are kept outside the repository.
