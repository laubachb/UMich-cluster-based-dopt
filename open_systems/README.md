# Open-data extension: does the cluster-local grade carry over to other systems?

Tests the manuscript's γ_cluster on three open data sets that a reader can obtain: Si GAP-18 (one element, 12 regimes),
the Si–O database of Erhard et al. (2024; SiO₂ composition, two elements, with the frames of its original active
learning marked) and the water database of Ibrahim et al. (2026; H and O, liquid and ice Ih–VII). Design and pass/fail criteria are fixed in [CRITERIA.md](CRITERIA.md) and were written before any
Si–O result. No new DFT is used.

## Why the test builds in a shift

On Si GAP-18 with a random in-distribution split (probe, 2026-10-08), the bulk basis under-represents several regimes
(diamond 0.11 of its proportional share, amorphous 0.07, vacancy 0.04), yet γ_bulk ranks their error as well as a basis
built within the true regime. The nitrogen held-out DFT-MD protocol of the manuscript shows the same. The manuscript's
gain appears when the candidates differ from the training data, so arm A makes a regime rare in training (a fraction
f = 2–25 % of its frames), and arm B grades the configurations that the original active learning added.

## Pipeline (run from this directory; `source ../env.sh` first)

| Step | Script | What it does |
|---|---|---|
| 1 | `01_prepare.py sio2` / `si_gap18` | Si–O: reads the Zenodo xyz (`$SCRATCH/cdopt_open_raw/database/`), keeps the SiO₂-composition config types (≤ 250 frames each), writes chimes_lsq inputs (2 elements, 228 coefficients). Si GAP-18: links the matrices prepared in `cluster-gamma-threshold` |
| – | `slurm/descriptors.cmd` | chimes_lsq on one spr node, `A.txt` → `A.npy` |
| 2 | `02_shift.py NAME` | arm A (shift-controlled): 4 fractions × 3 draws × every regime with ≥ 60 frames → `results/shift_NAME.csv` |
| 3 | `03_al_retro.py sio2` | arm B (active-learning retrospective) → `results/al_sio2.csv`, `results/al_sio2_rep.csv` |
| 4 | `04_criteria.py` | scores C1–C7 → `results/criteria.csv`, `results/shift_summary.csv` |
| 5 | `05_agreement.py NAME` | training-only agreement statistic A_R of C7 → `results/agreement_NAME.csv` |
| – | `slurm/analysis_dev.cmd`, `slurm/water_queue.sh` | skx-dev variants (one job per regime; feeder for the 4-job limit) |
| – | `slurm/analysis.cmd` | runs 2 or 3 on one spr node |
| 6 | `06_leverage.py NAME [--arm al] [--cluster]` | test R2: relative ridge leverage (predictive variance) of the candidates, same splits as arm A → `results/leverage_NAME.csv`, `results/leverage_al_sio2.csv` |
| 7 | `07_robustness.py` | scores R2-1 to R2-3 → `results/robustness_criteria.csv`, `results/robustness_by_regime.csv` |
| 8 | `08_min_rule.py NAME` | test M: minimum over cluster/regime bases (Shuang et al.) vs assignment → `results/minrule_NAME.csv` |
| 9 | `09_min_criteria.py` | scores M3 → `results/min_rule_criteria.csv` |
| 10 | `10_info_gain.py NAME [--cluster]` | test H: Bayesian information gain with regime noise (IG_het = RWV), partial pooling (IG_hier) → `results/infogain_NAME.csv` |
| 11 | `11_info_criteria.py` | scores H3 → `results/info_gain_criteria.csv` |
| 12 | `12_mlearn.py Cu Ge ... / --score` | test V: RWV confirmation on the untouched mlearn sets → `results/mlearn_EL.csv`, `results/mlearn_criteria.csv` |
| 13 | `13_advance.py NAME --methods soft+cal [--mlearn]` | test N: soft-membership grade, per-cluster conformal calibration (and the partition ensemble, dropped on the open data) → `results/advance_NAME.csv`, `results/advance_mlearn_EL.csv` |
| 14 | `14_advance_criteria.py [--mlearn METHOD]` | scores N1, N3b, picks the N4 candidate; N4 on mlearn → `results/advance_criteria_{open,mlearn}.csv` |
| 15 | `15_tda_features.py NAME / --nitrogen` | test T: per-atom persistent-homology (`tda`) and distance/angle (`ctrl`) descriptors from the positions → `work/NAME/{tda,ctrl}.npy` |
| 16 | `16_tda_partition.py NAME / --score` | test T: the manuscript's clustering on each representation, graded with the same bases → `results/tda_NAME.csv`, `results/tda_criteria.csv` |
| 18 | `18_ceiling.py NAME / --score` | test X: k-NN error-regressor ceilings of within-regime ranking → `results/ceiling_NAME.csv`, `results/ceiling_criteria.csv` |
| – | `slurm/robustness_dev.cmd` | one skx-dev job per MODE (nitrogen, open, al, nmin, min, nig, ig, nadv, adv, tda, tdan, tpart, ceil); see its header. `--export` splits on commas, so lists are space-separated in quotes and METHODS uses `+` |
| – | `slurm/mlearn_dev.cmd` | test V on one skx-dev node |

The nitrogen halves of these tests are `retrospective/11–15_*.py` and `md_audit/20–26_*.py` (see those READMEs).
Every test is pre-registered in [CRITERIA.md](CRITERIA.md) (criteria, predictions and recorded deviations, in the
order they were written); the scored results are the `results/*criteria*.csv` files.

`core.py` holds the shared code. Its MaxVol is the manuscript's algorithm with rank-one updates; `core.check_maxvol()`
reproduces the manuscript's γ_bulk on the nitrogen data to 4e-13. Bases are selected in the row space of their own
training rows (needed when a partition does not span all 228 Si–O columns); with full column rank this is the
manuscript's grade.

## Data provenance

- Si–O: Erhard, Rohrer, Albe, Deringer et al., "Modelling atomic and nanoscale structure in the silicon–oxygen system
  through active machine-learning" (2024). Zenodo 10.5281/zenodo.10419194, `database.zip`, CC-BY-4.0. SCAN, PAW GW
  potentials, 900 eV.
- Si GAP-18: Bartók, Kermode, Bernstein, Csányi, PRX 8, 041048 (2018), libAtoms/silicon-testing-framework.
- Water: Ibrahim, Lysogorskiy, Drautz, Piaggi, "Water phase diagram from a general-purpose atomic cluster expansion
  potential" (2026). Zenodo 10.5281/zenodo.18291736, CC-BY-4.0. revPBE-D3. No phase labels; regimes from O–O
  coordination (CRITERIA.md addendum). Reading the pickle needs `ase` (a venv in `$SCRATCH/cdopt_venv`).

## Results (2026-10-09)

Δ = AUROC of the partition grade − AUROC of γ_bulk for R's candidate atoms (10 % highest-error atoms as positives).

| Criterion | Si GAP-18 | SiO₂ | Water |
|---|---|---|---|
| C1 bulk basis under-represents a regime (f = 0.25) | pass (6) | pass (3) | **fail** (0) |
| C2 regime-basis gain larger at f = 0.02 than 0.25 | pass | pass (0.072 vs 0.065) | **fail** (not evaluable) |
| C3 regime basis, rare & under-represented | pass, +0.037 [0.019, 0.055] | pass, +0.089 [0.051, 0.126] | pass, +0.263 [0.224, 0.307] |
| C4 manuscript clustering, same scenarios | **fail**, −0.030 [−0.039, −0.020] | **fail**, −0.008 [−0.026, 0.011] | pass, +0.064 [0.028, 0.100] |
| C5 no loss > 0.05 where represented | pass | pass | pass (gains) |
| C6 active-learning analogue (vacancy) | – | pass (Δcluster +0.017, Δregime +0.090) | – |
| C7 A_R predicts Δ(cluster) | – | **fail as written**: ρ = −0.32 [−0.54, −0.04]; no scenario with A_R ≥ 0.8 | **fail as written**: ρ = −0.53 [−0.66, −0.29]; A_R 0.14–0.39 |

Reading:
- A basis selected within the true regime ranks error better than the bulk basis in most regimes of every data set
  (C3), when the regime is rare in training.
- The manuscript's clustering finds a useful partition only in the molecular system (water, as in nitrogen); in the
  network solids (Si, SiO₂) it does not (C4).
- The share of bulk-basis rows does not predict where γ_bulk fails: Si diamond has none and γ_bulk ranks it well;
  water's tetrahedral regime has its proportional share and γ_bulk is near random (AUROC 0.50–0.60).
- One basis per element (standard for multi-element linear potentials) is no better than γ_bulk in SiO₂ or water.
- The agreement statistic A behaves as predicted within data sets (negative correlation in both) and across them
  (water 0.14–0.39 and nitrogen 0.16–0.26 with gains; SiO₂ about 0.7 with none; Si 0.88 with a loss), but C7 fails as
  written because no scenario reached A ≥ 0.8 in the data sets where it was tested.

Notes on the runs: one skx-dev node failure (3584592) and one system cancellation (3584656) were resubmitted; the first
water C7 run crashed in its consistency check (merged and per-job files both matched) after computing all values, and
was rerun with the check fixed (3585201); its candidate sets match arm A for all 36 scenarios.

## Results of the robustness tests (2026-10-12/13)

Numbers are AUROC for the regime's 10 % highest-error atoms in the rare, under-represented scenarios (f ≤ 0.05,
representation < 0.5) unless stated; brackets are 95 % bootstrap intervals over scenarios. Full criteria and the
pre-registered predictions: [CRITERIA.md](CRITERIA.md); scored tables: `results/*criteria*.csv`.

| Test | Question | Result |
|---|---|---|
| R1 (nitrogen) | Do partitions from known conditions (density, state point) do what the clustering does? | Density partition +0.22–0.27 over γ_bulk in the molecular clusters (68 % of the clustering's gain). On the 5000 K MD the state-point partition ranks frames better than the clustering (Spearman 0.70 vs 0.47). |
| R2 | Does the fit's own predictive variance (leverage, `h`) rank error as well as the local γ? | Yes or better everywhere for ranking: h_bulk 0.79 / 0.84 / 0.86 (Si / SiO₂ / water) vs γ_bulk 0.73 / 0.69 / 0.58 and γ_regime 0.76 / 0.78 / 0.84. Exception: 5000 K MD frames (h_bulk 0.29, h_cluster 0.56, γ_cluster 0.47). |
| R3, S | Selecting frames by variance | h_cluster matches γ_cluster's 5000 K gain; MD collapse rate pooled over K: h_cluster 0.072, h_bulk 0.144 (γ_cluster 0.05, random 5000 K 0.16); at K ≤ 10 γ_cluster 0.02 vs h_cluster 0.11. |
| M | Assignment (ours) vs minimum over groups (Shuang et al., arXiv 2508.03405) | With true regimes, assignment wins on all three sets (+0.022 Si, +0.057 SiO₂, +0.096 water) and on nitrogen (0.80 vs 0.58). With imperfect clusters (water, Si) the minimum wins. |
| H | Information gain with regime-specific model error (IG_het = RWV) | Nitrogen molecular clusters 0.96 vs h_bulk 0.85; 5000 K MD not better than γ_cluster; +0.036 Si, −0.036 SiO₂, +0.016 water vs h_bulk. Noise-weighted MaxVol alone recovers most of γ_bulk's failure on the nitrogen candidates (0.45 → 0.81). |
| V | RWV on six untouched mlearn sets (Cu, Ge, Li, Mo, Ni, Si) | Passed all four criteria: vs h_bulk +0.024 [0.021, 0.028], vs γ_regime +0.049, vs γ_min +0.065, vs γ_bulk +0.088; positive in 6/6 elements. |
| N | Soft membership; per-cluster conformal calibration (γ_cal) | Soft: never worse than assignment (+0.011), loses to the minimum (−0.020); the only rule passing the selection test on nitrogen. γ_cal: pooled ranking across clusters 0.10 → 0.67 (nitrogen), coverage 0.88–0.93 at a 90 % target, +0.042 within regime on Si and water; N4 confirmation on mlearn pending. |
| T, X | Topological partition; ranking ceiling | Running (2026-10-13). |

## Transferring to another machine

What the repository does not contain, and where it is on Stampede3:

| Item | Location here | How to recreate elsewhere |
|---|---|---|
| Raw open data (Si–O, water) | `$SCRATCH/cdopt_open_raw/` | Download from the Zenodo records above (`database.zip`, `dataset_water_ACE_rev_PBE_D3.gzip`); set the paths at the top of `01_prepare.py` |
| Si GAP-18 and mlearn design matrices, forces, positions | symlinks in `work/si_gap18/` and `work/mlearn_*/` to `/work2/.../cluster-gamma-threshold/work/open/` | Copy those directories (about 1 GB) or rebuild: GAP-18 from libAtoms/silicon-testing-framework and mlearn from materialsvirtuallab/mlearn with the 42-term basis of `cluster-gamma-threshold/experiments/open_prep.py` |
| Si–O and water design matrices (`A.npy`, 228 columns) | `work/sio2/`, `work/water/` (git-ignored, ~3 GB) | `01_prepare.py NAME` then `slurm/descriptors.cmd` (chimes_lsq; `A.txt` → `A.npy`) |
| Per-scenario tables and job logs | `work/`, `results/` | Tables with results are the tracked `results/*.csv`; `work/*.out` are Slurm logs |
| ChIMES tools and LAMMPS | `../env.sh` → `cdopt_install_test/external/` | `install/install_chimes.sh` (see the root README). The build named in `env.sh` was removed on 2026-10-05; `md_audit/slurm/run_al_md_dev.cmd` points at the install-test build through `LMP_OVERRIDE` |
| Python | system `python3` 3.12 + `pip install --user ripser`; `$SCRATCH/cdopt_venv` (ase) for the water pickle | `pip install -r ../requirements.txt`; `ase` only for `01_prepare.py water` |
| Slurm | account `TG-CHM250015`; `skx-dev` (48 cores, 2 h, 2 running / 4 queued per user), `skx`, `spr` | Edit the `#SBATCH` headers in `slurm/*.cmd` and `../md_audit/slurm/*.cmd` |
