# Cluster-local D-optimality for active learning of machine-learned interatomic potentials

Code and data to reproduce the results of *"Cluster-Local D-Optimality Identifies Extrapolation That Bulk
Extrapolation Grades Miss in Machine-Learned Interatomic Potentials"* (Laubach, Lordi, Lindsey; in preparation).

## Motivation

Active learning of linear MLIPs (ChIMES, moment tensor potentials, SNAP) decides which configurations to label with
DFT using a D-optimal extrapolation grade γ. γ measures how far a candidate's descriptor lies outside the volume
spanned by a MaxVol-selected basis of training rows, and γ > 1 is the usual extrapolation threshold. The standard
implementation selects **one** basis from the whole training set (γ_bulk).

Training sets often mix physically distinct regimes of very different size. In the ChIMES nitrogen dataset, cold
molecular N₂ is a quarter of the data and hot, partly dissociated fluids are the rest. A basis selected to maximize
volume over the whole set sits mostly in the larger, more varied regime. As a result, a candidate can be far from
every training configuration of its own regime and still receive a low γ_bulk.

This repository tests a simple alternative, the cluster-local grade γ_cluster:
1. Cluster the per-atom force descriptors (the same rows that define the ChIMES model) without labels.
2. Select a MaxVol basis within each cluster.
3. Grade each atom against the basis of its own cluster.

## What is in this repository

| Study | Directory | Question |
|---|---|---|
| Retrospective | [`retrospective/`](retrospective/README.md) | On the original active-learning candidates (which have DFT forces), which grade flags and ranks the force error of a preliminary model? Is the gain the grade or the cluster label? Does it depend on the clustering? When should each grade be used? |
| MD audit | [`md_audit/`](md_audit/README.md) | When a DFT-only ChIMES model runs its own MD (900 frames checked with new DFT), which grade finds its errors? Which frames improve it most when added, and are the refit models stable in MD? Does γ warn of an instability before it happens? |

### Main findings

The argument in one sentence: a single bulk MaxVol basis is blind inside regimes it under-represents; γ_cluster
restores error ranking and precise flagging there, independently of the cluster label and of the clustering; where the
bulk basis is well resolved the two grades agree and γ_bulk flags more precisely; and data selected with γ_cluster
fixes the neglected regime fastest, including suppressing short-range collapse in MD.

- **Clustering finds the regimes.** Spectral clustering of the force descriptors separates molecular from
  dissociated nitrogen without temperature labels. The three molecular clusters hold 25% of the training atoms.
- **The mechanism is measurable.** The molecular clusters receive 0.7–1.4 of the 42 bulk-basis rows (0.2–0.4 of
  their fair share); the largest hot cluster alone receives 21.
- **Retrospective candidates, within the molecular clusters** (raw grades, no regressor, so the label plays no part):
  - γ_bulk flags 3–5 candidates per cluster and has no error ranking (Spearman ≈ −0.05).
  - γ_cluster flags 72–101 per cluster at 73–79% precision (share in the cluster's top-10% error), improves
    within-cluster Spearman by 0.26–0.31, and recovers 43% of the highest-error candidates at a 10% budget, against
    10% for γ_bulk and 12% for the cluster label alone.
  - In the largest (well-represented) cluster the two grades rank identically, and γ_bulk flags are more precise
    (87–91% vs 36–42%). Using γ_cluster only in under-represented clusters (a training-data-only rule) keeps the best
    of both.
  - Pooled across regimes, ranking is dominated by the regime itself: the cluster label alone reaches Spearman 0.62.
  - The result holds for every clustering tried (k = 4–8, three seeds).
- **In the model's own MD, in the reactive 5000 K fluid:**
  - γ_bulk flags 3 atoms. γ_cluster flags 275 (274 of them in the molecular clusters), with median force error 4.5×
    that of unflagged atoms; frame-level ranking Spearman 0.47 vs 0.08.
  - Adding γ_cluster-selected frames lowers the 5000 K error 2–3× faster than random selection, at the cost of a
    temporary error increase in the molecular regime of the 42-parameter model.
  - The refit models collapse (N–N pairs driven inside the inner cutoff) in 5% of 10 ps runs, against 45% for γ_bulk,
    31% for random and 16% for random frames drawn only from 5000 K (1,185 runs).
- **Negative result.** No grade warns of a collapse before it happens (AUROC ≈ 0.5 over 0.2–2 ps windows): the
  stability benefit comes from the data γ_cluster selects, not from detecting instabilities.

## Layout

```
common/          shared code: MaxVol + gamma + ranking metrics (dopt.py), paths and tool lookup (paths.py), file readers
data/            published ChIMES nitrogen dataset (xyzf, 6 MB) + provenance and DFT-protocol notes
install/         install_chimes.sh: clones and builds chimes_lsq, chimes_calculator and LAMMPS+chimesFF at pinned commits
retrospective/   scripts 01-08, fm_setup.in, results/ (tracked tables), slurm/ (example drivers)
md_audit/        scripts 00-15, models/, data/ (MD frames + DFT forces), dft_calib/, results/, slurm/
common/chimes_params.py  reads/writes the coefficients of a ChIMES params.txt (refit models for MD)
figures/         first-version figures (matplotlib) from retrospective/06_figures.py and md_audit/09_figures.py
requirements.txt pinned Python packages
```

The manuscript's final figures are drawn from the tracked result tables; `figures/` holds the first matplotlib versions
and is not updated for the later analyses (steps 07–08 and 10–15), whose numbers are in the `results/` tables.

`*/work/` directories hold large regenerable intermediates (design matrices, cluster labels, γ arrays, MD and DFT run
directories) and are git-ignored.

## Installation

```bash
python3 -m pip install -r requirements.txt           # Python 3.12 was used

export hosttype=UT-TACC                               # optional: loads modfiles/<hosttype>.mod of the ChIMES repos
bash install/install_chimes.sh                        # add --skip-lammps if you will not rerun the MD
source env.sh                                         # CHIMES_LSQ, CHIMES_CALCULATOR, LMP_CHIMES, MPIRUN, VASP_*
```

`install_chimes.sh` clones two public repositories into `external/` at the commits used for the paper:
- [chimes_lsq](https://github.com/laubachb/chimes_lsq-myLLfork) `a07329d7`
- [chimes_calculator](https://github.com/laubachb/chimes_calculator-myLLfork) `cff1feb5`

It then runs their `install.sh` scripts, including `etc/lmp/install.sh` for LAMMPS with `pair_style chimesFF`, and
writes `env.sh`. The ChIMES build scripts assume Intel oneAPI compilers and Intel MPI.

VASP is licensed separately and is only needed to regenerate the DFT labels. Set `VASP_GAM` (Γ-only executable) and
`VASP_POTCAR_N` (PAW_PBE N POTCAR) in `env.sh` if you want to rerun those steps.

## Reproducing the figures

Run everything from the repository root after `source env.sh`. On a SLURM machine, adapt the example drivers in
`*/slurm/`; they are written for TACC Stampede3.

**1. Retrospective study, Figs. 1–5** (about 1.5 h on one 48-core node; needs `chimes_lsq`):
```bash
python3 retrospective/01_descriptors.py      # chimes_lsq design matrix for 646 frames
python3 retrospective/02_cluster.py          # spectral clusters, candidate assignment, UMAP
python3 retrospective/03_errors.py           # DFT split + per-candidate force error of the preliminary model
python3 retrospective/04_gamma_ensemble.py   # gamma_bulk / gamma_cluster, 10 MaxVol solutions, regressors
python3 retrospective/05_controls.py 5       # control arms (label only, gamma_bulk + label, raw grades, ...)
python3 retrospective/06_figures.py          # figures/fig1-5
python3 retrospective/07_raw_and_hybrid.py   # raw grades, label control, flag precision, basis representation, switch rule
python3 retrospective/08_cluster_sensitivity.py   # k = 4-8 x 3 clustering seeds (needs md_audit step 6 for the MD part)
```

**2. MD audit, Fig. 6, from the tracked MD frames and DFT forces** (about 30 min; needs `chimes_lsq` and
`chimes_calculator`; requires step 1's `work/` outputs):
```bash
python3 md_audit/02_descriptors.py && python3 md_audit/03_gamma.py 10
python3 md_audit/06_errors.py && python3 md_audit/07_analysis.py && python3 md_audit/08_al_step.py
python3 md_audit/09_figures.py               # figures/fig6A, fig6B
python3 md_audit/10_balanced_split.py        # gamma-matched pool/test split
python3 md_audit/08_al_step.py --split balanced --save-coefs --random-sp 5000K_2.0gcc --n-maxvol 10 --out-tag _mv10
python3 md_audit/14_hybrid_md.py             # switch rule and hybrid grades on the MD audit
```

**3. MD stability of the refit models and the early-warning test** (LAMMPS; about 1,540 serial 10 ps runs, a few
node-hours; see `md_audit/README.md` §8b for the exact run sets):
```bash
python3 md_audit/11_al_md_setup.py balanced 1          # every refit model at 5000 K and 300 K, MD seed 1
sbatch -A <allocation> md_audit/slurm/run_al_md.cmd     # add -p spr --ntasks-per-node 112 on Sapphire Rapids
python3 md_audit/11_al_md_setup.py balanced_mv10 --seeds 1,2,3,4,5 --sp 5000K_2.0gcc \
    --select gamma_bulk:10,gamma_cluster:10,cluster_roundrobin:10,random:10,random_5000K:10 --out balanced_seeds
sbatch -A <allocation> --array=0-7 --export=ALL,SPLIT=balanced_seeds md_audit/slurm/run_al_md.cmd
python3 md_audit/11_al_md_setup.py balanced --seeds $(seq -s, 1 40) --sp 5000K_2.0gcc --select base:1 --out base_seeds
sbatch -A <allocation> --array=0 --export=ALL,SPLIT=base_seeds md_audit/slurm/run_al_md.cmd
for s in balanced balanced_seeds base_seeds; do python3 md_audit/12_al_md_analysis.py $s; done
python3 md_audit/13_al_md_seed_stats.py                # collapse-event rates and intervals
python3 md_audit/15_early_warning.py prep && sbatch -A <allocation> md_audit/slurm/run_early_warning.cmd
```
The runs in the paper were set up in stages (seed 1 for all models first, then extra seeds and MaxVol draws), so the
directory names differ slightly from the commands above; the model set and seeds are the same.

**4. Optional: regenerate the MD and DFT data** (LAMMPS; VASP, about 16 node-hours):
```bash
python3 md_audit/00_setup_md.py && sbatch md_audit/slurm/run_md.cmd && python3 md_audit/01_collect_frames.py
python3 md_audit/04_make_dft.py 8 && sbatch --array=0-7 md_audit/slurm/run_dft_array.cmd && python3 md_audit/05_collect_dft.py
```

A rerun of the MD with the same seeds is not guaranteed to be bitwise identical across machines or MPI layouts, so
it reproduces the MD-audit numbers statistically rather than exactly. Steps 1 and 2 above are deterministic; the
stability study (step 3) is statistical by design (rates over many seeds).

### Result → script → table

| Result | Script | Table(s) |
|---|---|---|
| Cluster composition, UMAP | `retrospective/02_cluster.py`, `06_figures.py` | `retrospective/results/cluster_composition.csv` |
| Flag counts and flag precision per cluster | `retrospective/04_gamma_ensemble.py`, `07_raw_and_hybrid.py` | `outliers.csv`, `flag_precision.csv` |
| Pooled ranking, label control | `retrospective/05_controls.py` | `controls.csv` (also `aggregate.csv` for the temporal splits) |
| Within-cluster ranking, recall vs budget (raw grades) | `retrospective/07_raw_and_hybrid.py` | `raw_by_cluster.csv`, `raw_recall.csv`, `raw_pooled.csv` |
| Bulk-basis representation, switch rule | `retrospective/07_raw_and_hybrid.py`; `md_audit/14_hybrid_md.py` | `bulk_basis_representation.csv`; `md_audit/results/hybrid_md_*.csv` |
| Clustering sensitivity | `retrospective/08_cluster_sensitivity.py` | `retrospective/results/cluster_sensitivity.csv` |
| MD audit: flagging and ranking | `md_audit/03_gamma.py`, `07_analysis.py` | `md_audit/results/audit_shared_t0_dft_train_{atom,frame,flags}.csv`, `flag_summary.csv` |
| One active-learning step | `md_audit/08_al_step.py`, `10_balanced_split.py` | `al_step.csv`, `al_step_balanced.csv`, `al_step_balanced_mv10.csv`, `balanced_split*.csv` |
| MD stability of the refits | `md_audit/11_al_md_setup.py`, `12_al_md_analysis.py`, `13_al_md_seed_stats.py` | `al_md_stability*.csv`, `al_md_seed_stats_{by_k,rules}.csv` |
| Early warning (negative) | `md_audit/15_early_warning.py` | `early_warning_{frames,hazard,alarm,profile}.csv` |
| DFT protocol | `md_audit/dft_calib/` | `dft_calibration.csv` |

## Verification

Checks built into or run alongside the pipeline:
- **MaxVol reimplementation** (`common/dopt.py`) agrees with the saved output of the original ChIMES implementation:
  Pearson 0.989 on γ_bulk, 0.82% vs 0.85% of candidates with γ > 1.
- **DFT protocol**: PBE + D2 at 1000 eV reproduces the published nitrogen forces to 0.005–0.008 eV/Å
  (`md_audit/results/dft_calibration.csv`).
- **Refit models**: a `params.txt` written by `common/chimes_params.py` gives chimes_calculator forces equal to the
  design-matrix prediction to 1e-10 kcal/mol/Å; the Python refit of the base model reproduces the forces of its
  published `params.txt` to 3e-5 kcal/mol/Å.
- **Reruns**: `08_al_step.py` on the seed split reproduces `results/al_step.csv` byte for byte; `--n-maxvol 10`
  reproduces the 5-draw models and rows exactly; `retrospective/07` reuses the MaxVol seeds and call order of step 04,
  so its γ are those of Figs. 2–5.
- **Descriptors of new frames** (`md_audit/15_early_warning.py`) match `md_audit/work/A.npy` to 1e-17 relative.
- **MD runs**: `slurm/run_al_md.cmd` counts a run as done only if LAMMPS finished its MD loop, and retries runs that
  hit a truncated input file on the shared filesystem; `12_al_md_analysis.py` reports any incomplete run.

## Data and citation

The nitrogen data are from the published ChIMES nitrogen model (Lindsey et al.,
[doi:10.1063/5.0157238](https://doi.org/10.1063/5.0157238)); see [`data/README.md`](data/README.md). The MD frames
and DFT forces in `md_audit/data/` were generated for this work.
