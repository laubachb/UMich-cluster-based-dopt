# Retrospective study: bulk vs cluster-local extrapolation grades on the ChIMES nitrogen candidates

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
figures go to `retrospective/results/` (not tracked; rebuilt by the scripts); figures go to `figures/`.

| Step | Script | What it does | Needs | Time |
|---|---|---|---|---|
| 1 | `01_descriptors.py` | chimes_lsq force-only design matrix for all 646 frames (`fm_setup.in`): 3 rows per atom × 42 columns → `A_atomic.npy`, frame and atom manifests | `CHIMES_LSQ`, `MPIRUN` | minutes on 48 cores |
| 2 | `02_cluster.py` | spectral clustering of DFT atoms, 15-NN assignment of candidate atoms, UMAP embedding | – | ~5 min |
| 3 | `03_errors.py` | 75/25 DFT frame split (seed 42) → `results/dft_holdout.json`; ridge fit on the training frames; per-atom candidate force error → `work/atom_errors.csv` | – | <1 min |
| 4 | `04_gamma_ensemble.py` | γ_bulk and γ_cluster for all candidates, 10 MaxVol solutions; flag counts; gradient-boosting error regressors (GroupKFold and temporal splits) → `results/{outliers,aggregate,by_cluster,recall}.csv` | – | ~1 h |
| 5 | `05_controls.py 5` | control arms (raw grades, label only, γ_bulk + label, …), 5 MaxVol solutions → `results/controls.csv` | – | ~30 min |
| 6 | `06_figures.py` | first-version Figs. 1–5 → `figures/`; cluster × state-point table → `results/cluster_composition.csv` | – | <1 min |
| 7 | `07_raw_and_hybrid.py` | the 10 MaxVol solutions of step 4 (same seeds and call order) scored without a regressor: raw grades, cluster-label control, flag precision per cluster, representation of each cluster in the bulk basis, and hybrid grades including the switch rule → `results/{raw_pooled,raw_by_cluster,raw_recall,flag_precision,bulk_basis_representation}.csv`, `work/gamma_retro.npz` | – | ~5 min |
| 8 | `08_cluster_sensitivity.py` | clustering redone for k = 4–8 × seeds 42, 0, 1; γ_cluster (5 solutions) scored on the manuscript's molecular candidates and on the 5000 K MD audit → `results/cluster_sensitivity.csv` | md_audit step 6 | ~15 min |

`slurm/run_retrospective.cmd` runs steps 1–6 on one node, and `slurm/run_story_analyses.cmd` runs steps 7–8 together
with `md_audit/14_hybrid_md.py` (TACC Stampede3 examples).

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
GroupKFold by frame, or train on the alc2 pool and test on alc3 or alc4. The cluster-label control of step 7 is the
out-of-fold mean error of the atom's cluster.

**Flag precision** (step 7) is the share of atoms with γ > 1 that are in their own cluster's top-10% error
(chance = 0.10).

**Bulk-basis representation and the switch rule** (step 7). For each MaxVol solution, the 42 rows of the bulk basis
are traced back to the clusters of their atoms. A cluster's representation is its share of basis rows divided by its
share of training rows (1 = fair share). The switch rule grades atoms of clusters with representation < 1 by
γ_cluster and all others by γ_bulk. It uses only the training data. Other hybrids scored: max(γ_bulk, γ_cluster) and
the maximum of the two grades' percentile ranks.

## Results and how to read them

| Result | Content | Headline |
|---|---|---|
| Clusters | UMAP of DFT atoms by cluster and state point | Clustering separates molecular (c0, c3, c4) from dissociated (c1, c2, c5) nitrogen without temperature labels |
| Flags | Candidates with γ > 1 per cluster, and their precision | Minority clusters: γ_bulk flags 3–5 atoms each, γ_cluster flags 72–101 that γ_bulk misses, at 73–79% precision. Majority clusters: γ_bulk flags more (c1: 193) and more precisely (87–91%, vs 36–42% for γ_cluster-only flags) |
| Pooled ranking | Raw grades and regressors with and without the cluster label | Raw γ_bulk 0.09, raw γ_cluster 0.12 (Spearman); label alone 0.62, γ_bulk + label 0.65, γ_cluster + label 0.67. **Pooled ranking is dominated by the regime label** |
| Within-cluster ranking | Raw γ_cluster − raw γ_bulk inside each cluster (label constant) | Minority clusters +0.26 to +0.31 Spearman, +0.34 to +0.37 AUROC; c1 −0.01 ± 0.03; c2, c5 +0.08. Raw γ_bulk has no ranking inside the minority clusters |
| Budget | Recall of the top-10% error minority candidates vs labelling budget | At 10%: γ_cluster 0.43, γ_bulk 0.10, cluster label 0.12, random 0.10 |
| Representation, switch | Bulk-basis rows per cluster; flag precision of the switch rule | Minority clusters get 0.7–1.4 of 42 rows (0.18–0.41 of fair share), c1 gets 21. The switch rule is as precise as γ_cluster in the minority clusters (0.77) and as γ_bulk elsewhere (0.87) |
| Clustering sensitivity | 15 clusterings (k = 4–8 × 3 seeds) | γ_cluster beats γ_bulk in all: minority Spearman 0.14–0.34 vs −0.07, recall 0.22–0.50 vs 0.08. Weaker at k = 4–5, plateau from k = 6; the seed hardly matters |

Supported claim: γ_bulk does not rank or flag error within regimes the bulk basis under-represents, and γ_cluster
does, independently of the cluster label and of the clustering; where the bulk basis is well resolved the grades rank
alike and γ_bulk flags more precisely. The first-version figures in `figures/` plot regressor scores (steps 4–5); the
revised analysis above uses the raw grades of step 7.

## Notes

- **MaxVol reimplementation.** The original γ code used in the ChIMES workflow was unavailable, and `common/dopt.py`
  reimplements rectangular MaxVol. Its γ_bulk agrees with the saved output of the original (Pearson 0.989; 0.82% vs
  0.85% of candidates with γ > 1). Across solutions, Spearman(γ_bulk, error) ranges from −0.01 to 0.27, which is why
  all results are averaged.
- The switch rule is scored here and on the MD audit (`../md_audit/14_hybrid_md.py`) but has not been used as a
  selection rule in the active-learning step.
- The earlier Euler-characteristic clustering variant, the selection → refit experiments ("Mode I") and other
  exploratory analyses are not part of this pipeline and are kept outside the repository.
