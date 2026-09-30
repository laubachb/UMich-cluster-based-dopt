# MD audit: do bulk and cluster-local γ find a real MLIP's errors in its own MD?

This directory holds the MD part of the study. It tests the retrospective result
(`../retrospective/`) on a trained ChIMES-N model running its own molecular dynamics:

1. **Flagging.** When the model runs its own MD, do bulk γ and cluster γ (γ_bulk, γ_cluster) point at the atoms and frames
   where it is actually wrong? "Wrong" means the per-atom force error against new DFT single points.
2. **Active learning.** If DFT frames from that MD are chosen by γ_bulk, by γ_cluster or at random, added to the
   training set and refit, which choice lowers held-out error fastest, and in which regime?
3. **Stability.** Are the refit models stable when they run their own MD, judged by the conserved quantity of
   Langevin dynamics and by short-range collapse (§8b)?
4. **Switch rule and early warning.** Does using γ_cluster only in clusters the bulk basis under-represents keep the
   best of both grades, and does γ rise before a collapse (§8c)?

All numbers below are read from the tracked result files listed in the [file map](#file-map-and-reproduction).

## Summary

- **The DFT-only ChIMES-N model leaves its training coverage within the first picosecond.** 51–56% of atom-frames at 300 K and
  2000 K get γ_cluster > 1, against 16–19% for γ_bulk. In every run and state point, no atom is robustly flagged by
  γ_bulk alone.
- **Both scores track the true DFT error**, but where each one helps depends on the regime:
  - In the molecular regime (300 K and 2000 K) the two rank atom errors about equally
    (Spearman 0.55 vs 0.53 at 300 K; 0.40 vs 0.40 at 2000 K).
  - In the reactive regime (5000 K, 2.0 g/cc), γ_cluster ranks errors clearly better. Atom level: 0.27 [0.26, 0.29]
    vs 0.15 [0.13, 0.16]. Frame level: 0.47 [0.36, 0.55] vs 0.08 [−0.03, 0.19].
  - At 5000 K, γ_bulk flags 3 atoms and γ_cluster flags 275 more. Those 275 have a median force error of
    4.3 eV/Å against 0.96 eV/Å for unflagged atoms, and 59% of them fall in the worst 10%.
- **Active learning.** Frames chosen by γ_cluster lower the 5000 K held-out error 2–3× faster than random frames. The
  improvement from 1.257 eV/Å is 0.13–0.16 eV/Å for γ_cluster vs 0.065 ± 0.014 for random; γ_bulk gives only 0.03.
  This has a price in the 42-parameter model: the first 5–10 γ_cluster frames make 300 K and 2000 K error *worse*, and
  error on 30 held-out DFT frames rises by 0.025 eV/Å. γ_bulk puts its budget almost entirely into 2000 K frames and
  performs about the same as random there.
- **Stability (§8b).** At 5000 K the refit models collapse (N–N pairs inside the inner cutoff, jump of the conserved
  quantity) in 5% of 10 ps runs for γ_cluster, 10% for round-robin, 16% for random 5000 K frames, 31% for random,
  45% for γ_bulk and 33% for the base model (50 runs per rule and K, 1,185 runs). At 300 K nothing collapses, but the
  base model over-condenses the liquid until molecular frames are added.
- **Switch rule (§8c).** The same three molecular clusters are under-represented in this model's bulk basis. Grading
  them with γ_cluster and the rest with γ_bulk matches γ_cluster at 5000 K (frame Spearman 0.47) and restores
  γ_bulk's precise flags at 300 and 2000 K.
- **No early warning (§8c).** No grade separates the frames just before a collapse from the rest (AUROC ≈ 0.5).
- **Interpretation.** In real MD, bulk normalisation is blind to the regime the model reaches but was never trained to
  resolve internally (here the dissociating 5000 K fluid). Cluster-local γ finds that regime and fixes it fastest. It
  is not better everywhere, and a model with limited capacity pays for the added data in other regimes.

---

## 1. Background

The retrospective study (Figs. 2–5) shows that γ_bulk does not rank error within under-represented regimes of the
ChIMES-N candidate set, while γ_cluster does. Its control arms (`../retrospective/05_controls.py`,
`../retrospective/results/controls.csv`) show that the within-cluster (Fig. 4) and budget (Fig. 5) gains come from the
cluster-local grade, not from the cluster label. The practical question is whether this matters when a model runs MD.
Those candidates were collected by earlier models during the original campaign, so this study runs MD with a model
and labels that MD with new DFT. Nitrogen cells are 64 atoms, so a DFT single point costs about 20–65 s.

## 2. The models

All models are ChIMES-N force fields in `models/<model>/` (`params.txt` and `train_meta.json`, which lists the exact
training frames).

| Setting | Value |
|---|---|
| Bodies / orders | 2-body + 3-body Chebyshev, orders 12 / 5 (`CHEBYSHEV 12 5 0`) |
| Cutoffs | 2-body r_c = 8.0 Å, 3-body 5.0 Å, r_min = 0.86 Å |
| Distance transform | Morse, λ = 1.09 Å |
| Smooth cutoff | Tersoff, f_O = 0.75 |
| Parameters | 42 |
| Fit | force-only (FITENER/FITSTRS false), unweighted; `chimes_lsq` design matrix, `chimes_lsq.py --algorithm svd --eps 1e-8` |

| Model | Training set |
|---|---|
| `shared_t0_dft_train` (**the "DFT-only model", the subject of this study**) | 90 DFT frames: 15 from each of the 6 DFT state points (300 K 1.0, 2000 K 1.0, 5000 K 2.0, 6000 K 2.5, 7000 K 3.1, 8000 K 4.5 g/cc). The other 30 DFT frames are the holdout. Training force RMSE 26.59 kcal/mol/Å (1.153 eV/Å). |
| `bulk_t3_k30`, `cluster_t3_k30`, `random_t3_k30`, `stratified_t3_k30` | t0 + 90 active-learning candidate frames (30 from each of ALC pools 2, 3 and 4), chosen by the named rule; 180 frames in total. These come from an earlier selection → refit experiment that is not part of this repository; they are provided as fixed inputs and are used only for the flag-rate comparison (`results/flag_summary.csv`). |

**Checks on the DFT-only model**
- *Exact refit.* A Python least-squares refit on the same 90 frames (`08_al_step.py`) reproduces its
  `params.txt` to within 3 × 10⁻⁵ kcal/mol/Å on an MD frame, with the same training RMSE (26.591).
- *N₂ dimer* (chimes_calculator, 17 Å box). r_eq = 1.106 Å and k = 2391 N/m, against about 2290 N/m for PBE N₂. This
  predicts a classical bond-length spread of 0.0132 Å at 300 K, so the model's bond stiffness is physical. The t3
  models agree (r_eq 1.095–1.097 Å, k 2357–2447 N/m).

The cluster labels are those of the retrospective study (`../retrospective/02_cluster.py`: k = 6, on standardized
per-atom [fx|fy|fz] descriptors of the 10,560 DFT atoms). The minority (molecular) clusters are c0, c3 and c4. Per-cluster atom counts in the DFT-only
model's training set are [716, 3500, 1226, 630, 656, 1192].

## 3. MD simulations

LAMMPS with `pair_style chimesFF` (built by `../install/install_chimes.sh`, `$LMP_CHIMES`), `units real`, 64 N
atoms in cubic periodic cells. Each run starts from frame 0 of the DFT trajectory at its state point:

| State point | Box edge |
|---|---|
| 300 K and 2000 K, 1.0 g/cc | 11.41939 Å |
| 5000 K, 2.0 g/cc | 9.06400 Å |

### 3a. The original MD had a thermostat artifact (not used for results)

The first MD runs of this study used Nosé–Hoover NVT with T_damp = 20 fs, a 0.2 fs timestep, 5 ps and a dump
every 100 fs (51 frames). In those runs the N₂ bond stretch drains:

| 300 K, DFT-only model | Bond-length s.d. |
|---|---|
| DFT reference frames | 0.0148 Å |
| MD at t = 0 | 0.0156 Å |
| MD at 0.1–0.3 ps | 0.0079 Å |
| MD after 2 ps | 0.0016 Å (an effective stretch temperature of about 4 K) |

At 2000 K the spread falls from 0.034 to 0.024 Å. The model's dimer stiffness is correct (§2), so this is a
sampling artifact: T_damp is close to the ~14 fs N₂ stretch period, and the stretch barely exchanges energy with other
modes. γ flags in these runs grow as the bonds cool, so the runs were **redone with Langevin dynamics** before any DFT
was spent. Those runs are not part of this repository.

### 3b. Langevin MD (used for all results)

`00_setup_md.py` writes `work/md/<model>__<statepoint>__s<seed>/{in.lammps,data.in,params.txt}`:

| Setting | Value |
|---|---|
| Integration | `fix nve` + `fix langevin T T 100.0 <seed> zero yes` (100 fs damping) |
| Timestep, length | 0.2 fs, 50,000 steps = 10 ps |
| Output | dump every 500 steps (100 fs), `%.8f` coordinates, 101 frames per run including t = 0 |
| Velocities | `velocity all create T <s>7<s>1 dist gaussian mom yes rot yes` |
| Langevin seed | `<s>3<s>9` |
| Runs | 5 models × {300 K, 2000 K} × seeds {1, 2, 3}, plus the DFT-only model × 5000 K × seeds {1, 2, 3} = 33 runs |
| Hardware | 16 MPI ranks per run, 3 runs at a time on one skx node, about 204 s per run |

With Langevin dynamics the N₂ bond statistics match DFT: at 300 K, s.d. 0.0138 Å (DFT 0.0148); at 2000 K,
0.0378 Å (DFT 0.0338).

## 4. Descriptors and γ

**Descriptors** (`01_collect_frames.py`, `02_descriptors.py`). Every saved frame (3,333 frames, 213,312 atoms) is
written to `data/traj.xyzf.gz` with zero forces; the design matrix depends only on geometry. `chimes_lsq` runs with
the retrospective study's `fm_setup.in`, giving three rows per atom (fx, fy, fz) × 42 columns. *Check:* a t = 0 frame
(a DFT frame) of the first Nosé–Hoover runs matched the retrospective `A_atomic.npy` rows to 5 × 10⁻⁴ relative error,
which is the rounding of the 6-significant-digit coordinates in those dumps.

**Cluster assignment.** The same as the paper. The concatenated per-atom descriptors are standardized with a
StandardScaler fit on the DFT atoms, and each MD atom takes a 15-nearest-neighbour, distance-weighted vote of the
DFT atoms' spectral labels.

**γ** (`03_gamma.py`). Scored against **the training set of the model that ran that MD**, rebuilt exactly from
`models/<model>/train_meta.json`:
- *γ_bulk:* MaxVol on all of that model's training rows.
- *γ_cluster:* MaxVol on that model's training atoms with the same cluster label.
- *MaxVol:* the random-restart rectangular MaxVol of the retrospective study (`../common/dopt.py`),
  rank 42, tolerance 1.01, 10 independent solutions (seeds 300–309).
- *Per-atom γ:* the maximum over the three component rows of max |a·Â⁻¹|, averaged over the 10 solutions.
- *Robust flag:* γ > 1 in at least 80% of the solutions. "Cluster-only" additionally requires γ_bulk ≤ 1 in every
  solution.

## 5. DFT single points

**Calibration** (`dft_calib/make_calib.py`, `dft_calib/compare.py` → `results/dft_calibration.csv`). The dataset
README says only "allPBE". Frame 5 of
four DFT files was recomputed under four candidate settings and compared with the stored forces (converted from
Ha/bohr × 51.422 to eV/Å):

| Setting | 300 K | 2000 K | 5000 K | 8000 K (120 atoms) |
|---|---|---|---|---|
| PBE, ENCUT 700 | 0.0194 | 0.0311 | 0.0420 | 0.0361 |
| PBE + D2, ENCUT 700 | 0.0205 | 0.0244 | 0.0273 | 0.0169 |
| PBE, ENCUT 1000 | 0.0150 | 0.0192 | 0.0283 | 0.0328 |
| **PBE + D2, ENCUT 1000** | **0.0076** | **0.0062** | **0.0059** | **0.0048** |

Values are force RMSE against the dataset in eV/Å. For scale, the force RMS is 1.6–4.5 eV/Å.

**Protocol used** (`04_make_dft.py`):

| Setting | Value |
|---|---|
| Code | VASP 5.4.4, Γ-only build (`$VASP_GAM`) |
| Pseudopotential | PAW_PBE N (08Apr2002) |
| Functional | PBE + D2 (`IVDW = 1`) |
| Basis | `ENCUT = 1000`, `PREC = Accurate`, `LREAL = .FALSE.` |
| Smearing | `ISMEAR = -1` (Fermi) with `SIGMA = k_B·T`: 0.025852, 0.172347 and 0.430867 eV at 300, 2000 and 5000 K |
| SCF | `EDIFF = 1E-06`, `NELM = 200`, `NELMIN = 4`, `ALGO = Normal` |
| Other | `ISYM = 0`, `IBRION = -1`, `NSW = 0`, Γ-point 1×1×1 k-mesh |
| Parallel | `NCORE = 8`, 48 MPI ranks per frame on one skx node |

**What was labelled.** Every frame with t > 0 from the DFT-only model's 9 Langevin runs: 3 state points × 3 seeds ×
100 frames = 900 frames (57,600 atoms). All converged. The cost was about 65 s per frame on average (8-task array,
about 2 h wall; `slurm/run_dft_array.cmd`). `05_collect_dft.py` stores the forces of all 900 frames in
`data/dft_forces.npz` (1.1 MB, tracked), so the analysis can be reproduced without VASP.

## 6. Error definition

`06_errors.py` computes the following for every DFT-labelled atom:
- **F_model:** the DFT-only model's forces from chimes_calculator with its `params.txt`, converted from kcal/mol/Å to
  eV/Å (× 0.0433641).
- **F_DFT:** `data/dft_forces.npz` (the last `TOTAL-FORCE` block of each OUTCAR). It includes D2, like the training labels, so the comparison
  is like for like.
- **err = |F_model − F_DFT|** (vector norm, eV/Å).

The active-learning metric in §8 is the **component** RMSE, roughly err/√3, in eV/Å.

Error level of the DFT-only model on its own MD (RMS of err, then median err):

| State point | RMS of err (eV/Å) | Median err (eV/Å) | RMS DFT force (eV/Å) |
|---|---|---|---|
| 300 K | 0.74 | 0.70 | 2.23 |
| 2000 K | 1.02 | 0.82 | 5.46 |
| 5000 K | 2.20 | 1.01 | 8.04 (max err 14.7) |

## 7. Result 1: does γ find the model's own MD errors?

`07_analysis.py` → `results/audit_shared_t0_dft_train_{atom,frame,flags}.csv`; figure
`../figures/fig6A_md_flagging.png`. Brackets are 95% bootstrap intervals over frames.

**Atom-level Spearman ρ(γ, err)** (19,200 atoms per state point):

| | γ_bulk | γ_cluster | Random |
|---|---|---|---|
| 300 K | **0.549** [0.531, 0.567] | 0.534 [0.522, 0.551] | −0.004 |
| 2000 K | 0.395 [0.383, 0.409] | 0.401 [0.384, 0.413] | 0.014 |
| 5000 K | 0.147 [0.134, 0.162] | **0.272** [0.256, 0.285] | 0.007 |

**Other atom-level metrics:**

| | AUROC, worst 10% (bulk / cluster) | Recall of worst 10% at a 10% budget (bulk / cluster) |
|---|---|---|
| 300 K | 0.795 / 0.758 | 0.380 / 0.347 |
| 2000 K | 0.536 / 0.549 | 0.135 / 0.122 |
| 5000 K | 0.637 / **0.694** | 0.191 / **0.277** |

**Within-cluster Spearman (γ_bulk / γ_cluster):**

| | c0 | c4 | Other clusters |
|---|---|---|---|
| 300 K | 0.57 / **0.65** | 0.54 / **0.62** | equal (c2, c3, c5) |
| 5000 K | 0.19 / **0.47** | 0.17 / **0.44** | γ_cluster ahead in c2, c3 and c5 (0.07–0.13 → 0.25–0.36) |

**Frame-level Spearman ρ(frame-max γ, frame force RMSE)** (300 frames per state point; the frame is what gets sent to
DFT):

| | γ_bulk | γ_cluster |
|---|---|---|
| 300 K | 0.196 [0.09, 0.29] | 0.206 [0.10, 0.31] |
| 2000 K | 0.171 [0.04, 0.29] | 0.228 [0.10, 0.34] |
| 5000 K | 0.077 [−0.03, 0.19] | **0.466** [0.36, 0.55] |

**Robust flags and the true error of flagged atoms:**

| | Unflagged: n, median err | γ_bulk > 1: n, median err, share in worst 10% | γ_cluster > 1 only: n, median err, share in worst 10% |
|---|---|---|---|
| 300 K | 4010, 0.46 | 607, 1.03, 47% | 1597, 0.70, 5% |
| 2000 K | 4910, 0.56 | 707, 1.00, 15% | 1922, 0.78, 13% |
| 5000 K | 17468, 0.96 | **3**, 7.9, — | **275, 4.28, 59%** |

**How much of each run is flagged** (`results/flag_summary.csv`, all 33 runs, fraction of atom-frames with
γ > 1): the DFT-only model is flagged at 51–56% by γ_cluster and 16–19% by γ_bulk at 300 and 2000 K, and 3.1% vs
0.1% at 5000 K. The four t3 models, which have more training data, are flagged at 30–43% vs 6–16%. No run has a
robust bulk-only flag.

In time, for the DFT-only model's Langevin runs, the fraction of atoms with γ_cluster > 1 goes as follows:

| Window | 300 K | 2000 K | 5000 K |
|---|---|---|---|
| 0.1–0.3 ps | 22% | 2% | 2% |
| 0.4–1.0 ps | 59% | 56% | 2% |
| After 1 ps | 54–58% | 59–60% | 2% |

γ_bulk lags. At 300 K it goes 0% → 7% → 15%; at 2000 K, 0% → 2% → 12–22%. So γ_cluster registers the drift out of
training coverage within the first picosecond, before γ_bulk. At 5000 K the flag rate stays flat at about 2%.

**Reading of result 1**
- In the molecular regime both scores rank errors similarly. Used as a γ > 1 threshold, γ_bulk is more *precise*
  (few flags, with about 2× the unflagged error) and γ_cluster is more *sensitive* (about 3× more flags, with about
  1.5× the unflagged error). γ_cluster > 1 fires on half the atoms, so 1 is not a calibrated threshold for it; ranking
  is the right use.
- In the reactive 5000 K fluid, the bulk basis, dominated by the populous high-T clusters, is saturated, and γ_bulk
  barely ranks the model's worst atoms or frames. γ_cluster ranks them well and flags a small, high-precision set of
  very bad atoms that γ_bulk misses.
- This is a different regime from where figs 4–5 place the gain (hidden clusters in the candidate set). The common
  mechanism is that the gain appears wherever the model's MD reaches configurations whose within-regime variation the
  bulk-normalized basis cannot resolve.

## 8. Result 2: one active-learning iteration

`08_al_step.py` → `results/al_step.csv`; figure `../figures/fig6B_md_al_step.png`.

**Design**
- *Base:* the DFT-only model, refit exactly as described in §2.
- *Pool:* 600 DFT-labelled Langevin MD frames (seeds 1–2, all three state points).
- *Test:* the 300 frames of seed 3 (per state point) plus the 30 held-out DFT frames.
- *Budgets:* K = 2, 5, 10, 20 and 40 frames are added with their DFT forces (eV/Å × 23.0605 → kcal/mol/Å), and the
  model is refit with the same unweighted least squares.
- *Arms:*
  - **γ_bulk / γ_cluster:** pool frames ranked once by frame-max γ against the base training set; 5 independent
    MaxVol solutions.
  - **γ_cluster, round-robin:** each frame is keyed to the cluster of its highest-γ_cluster atom, and the top frame of
    each cluster is taken in turn (the per-cluster selection rule); 5 MaxVol solutions.
  - **Random:** 20 draws.
  - A greedy variant that recomputes γ after each added frame is implemented (`--greedy`) but was not run.

**Held-out force RMSE** (component, eV/Å; mean ± s.d.):

| Arm, K | 300 K MD | 2000 K MD | 5000 K MD | 30 DFT holdout |
|---|---|---|---|---|
| base, 0 | 0.436 | 0.585 | 1.257 | 1.142 |
| random, 10 | 0.258 ± 0.017 | 0.463 ± 0.009 | 1.237 ± 0.008 | 1.141 |
| random, 40 | 0.205 ± 0.005 | 0.402 ± 0.007 | 1.192 ± 0.014 | 1.145 |
| γ_bulk, 10 | 0.254 | 0.454 | 1.245 | 1.143 |
| γ_bulk, 40 | 0.205 | **0.392** ± 0.011 | 1.230 | 1.145 |
| γ_cluster, 10 | 0.483 | 0.613 | 1.183 | 1.152 |
| γ_cluster, 40 | 0.213 | 0.414 | 1.126 ± 0.009 | 1.166 |
| round-robin, 10 | 0.522 ± 0.062 | 0.635 ± 0.046 | 1.197 | 1.150 |
| round-robin, 40 | 0.316 ± 0.058 | 0.465 ± 0.039 | **1.102** ± 0.009 | 1.169 |

**What each arm picks** (draw 0): γ_bulk takes almost only 2000 K frames (38 of 40). γ_cluster starts with 5000 K
frames (all of the first 10) and then spreads out (18 / 12 / 10 at K = 40). Round-robin is 35 of 40 at 5000 K.

**Reading of result 2**
- At 5000 K, γ_cluster selection lowers error 2–3× faster than random: a 0.13–0.16 eV/Å improvement against
  0.065 ± 0.014, where the random spread is about 0.014. γ_bulk is worse than random there (0.03).
- At 300 and 2000 K, γ_bulk is about equal to random, or slightly better at 2000 K and K = 40 (0.392 vs
  0.402 ± 0.007). γ_cluster's early 5000 K frames make these regimes worse than the base model (e.g. 2000 K
  0.613 vs 0.585 at K = 10), and error recovers only by K = 40. The 30 DFT holdout frames get slightly worse under
  both γ_cluster arms (+0.025 eV/Å).
- The trade-off is a capacity effect. One 42-parameter, force-only, unweighted fit cannot absorb reactive
  high-force data without moving the molecular regime. A per-regime weighting, a larger basis, or a greedy batch
  that updates γ after each frame are the natural next checks.
- Random selection stays a strong overall baseline, consistent with an earlier selection-and-refit experiment on the
  retrospective candidates (not included in this repository).

## 8b. Balanced split and MD stability of the refits

**γ-balanced split** (`10_balanced_split.py`). The seed split (pool = seeds 1–2, test = seed 3) leaves pool and test with
different γ distributions at 300 K (KS 0.55 on frame-max γ_bulk, 0.54 on γ_cluster), because seed 1 runs differently.
The balanced split holds out 10 of 30 one-picosecond blocks per state point (same 200/100 sizes), chosen to minimise the
largest pool-vs-test KS statistic over γ features only (frame-max γ, fraction of atoms with γ > 1, per-atom γ; never the
error). Largest KS: 0.07 (300 K), 0.055 (2000 K), 0.05 (5000 K). Rerunning step 8 on it
(`results/al_step_balanced.csv`, `../manuscript_figures/main/gnuplot/output/fig6B_md_al_step_balanced.png`) gives the
same result: from 1.233 eV/Å at 5000 K, 40 frames lower the error by 0.115 (γ_cluster) and 0.154 (round-robin) against
0.064 (random) and 0.023 (γ_bulk); at 300 K γ_cluster first rises, then returns to the random error at K = 40.
The seed-split rerun reproduces `results/al_step.csv` exactly.

Why γ_cluster raises 300 K error first: its first 10 frames are all 5000 K. A control refit adding 10 random 5000 K
frames gives 300 K 0.445 (base 0.431), 10 random 2000 K frames give 0.234 (the γ_bulk result). Down-weighting the added
5000 K rows removes both the 300 K cost and the 5000 K gain, so the trade-off is capacity of the unweighted 42-term fit.

**MD stability** (`11_al_md_setup.py`, `slurm/run_al_md.cmd`, `12_al_md_analysis.py`). All 176 balanced-split refits
(base + 3 γ rules × 5 K × 5 MaxVol draws + random × 5 K × 20 draws) ran 10 ps at 5000 K and 300 K with the Fig. 6
Langevin protocol and `tally yes`, so `econserve = etotal + ecouple` is conserved; the base model also ran 20 seeds per
state point (`work/al_md/base_seeds`). All 392 runs completed. Draft figure: `fig6C_md_stability.png`.
- *300 K:* every run conserves econserve to ≤ 0.034 meV/atom and keeps all 32 N₂ intact. But the base model and the
  early γ_cluster/round-robin refits condense: nearest N₂-centre distance 2.78–2.80 Å against 3.34 Å (3.26–3.47) in
  DFT-MD, with 6 other centres within 3.3 Å instead of 0.5. Refits given 2000 K frames approach DFT (γ_bulk K = 40:
  3.19 Å; random 3.13 Å); γ_cluster recovers at K = 40 (3.10 Å). Same ordering as the 300 K force error.
- *5000 K:* runs either conserve econserve to ~1 meV/atom (median 0.6) or show a collapse event: an N–N pair driven
  inside the 0.86 Å inner cutoff (down to 0.36 Å) with an econserve jump of 10²–10⁴ meV/atom.
- *Seeds and control arm* (`08_al_step.py --random-sp 5000K_2.0gcc` adds `random_5000K`: uniform draws from the 5000 K
  pool frames only; `11_al_md_setup.py ... --out balanced_seeds`; `13_al_md_seed_stats.py`). At 5000 K every rule has
  10 models per K, each run at seeds 1–5 (50 runs per point): γ rules = MaxVol draws 0–9 (draws 5–9 from
  `08_al_step.py --n-maxvol 10 --out-tag _mv10`, which reproduces draws 0–4 exactly), random and `random_5000K` = draws
  0–9. The base model ran 40 seeds. All 1,185 runs at 5000 K completed. Event rate pooled over K (95% Wilson), and
  difference to `random_5000K` with a 95% interval from a bootstrap over models (a model's seeds move together):

  | Rule | Events / runs | Rate | vs random 5000 K only |
  |---|---|---|---|
  | base (40 seeds) | 13 / 40 | 0.33 [0.20, 0.48] | |
  | γ_bulk | 112 / 250 | 0.45 [0.39, 0.51] | +0.29 [0.21, 0.37] |
  | random | 77 / 250 | 0.31 [0.25, 0.37] | +0.15 [0.07, 0.23] |
  | random, 5000 K only | 39 / 250 | 0.16 [0.12, 0.21] | |
  | γ_cluster | 12 / 250 | 0.05 [0.03, 0.08] | −0.11 [−0.17, −0.05] |
  | γ_cluster, round-robin | 25 / 250 | 0.10 [0.07, 0.14] | −0.06 [−0.12, +0.01] |

  At K ≤ 10, where every γ_cluster frame is a 5000 K frame: γ_cluster 3/150 = 0.02 against random 5000 K frames
  31/150 = 0.21 (difference −0.19 [−0.27, −0.11]). γ_cluster's rate rises slowly with K (0.00, 0.02, 0.04, 0.06, 0.12
  at K = 2–40) as its selection moves on to 300/2000 K frames; at K = 40 it is 6/50 = 0.12 against 5/50 = 0.10 for
  random 5000 K frames. So adding 5000 K data reduces collapses, and the 5000 K frames γ_cluster ranks highest reduce
  them about as much again, per frame. Round-robin is not distinguishable from γ_cluster or from the control: its
  K = 2 models collapse at 0.26, then 0.02–0.10.
  Filesystem note: with ~50–100 runs starting at once on /work2, LAMMPS occasionally read a truncated data.in or an
  empty in.lammps; `slurm/run_al_md.cmd` now counts a run as done only if its log has "Loop time" and retries otherwise.
  Six base seeds and three refit runs were rerun on the spr partition (same LAMMPS build).
  The force error shows the same crossover: at K = 10, 5000 K RMSE 1.166 (γ_cluster) vs 1.186 (random 5000 K); at
  K = 40, 1.118 vs 1.090. Figures: `fig6C_md_stability.png` (event rate vs K; 300 K packing), `figS_md_econs_strip.png`
  (per-run excursions, seed 1).
- *Caveat:* 10 ps per run catches early failures only; each point is 50 runs from 10 models, so intervals come from
  the model-level bootstrap, not the run count.

## 8c. Switch rule and early warning

**Switch rule on the MD audit** (`14_hybrid_md.py` → `results/hybrid_md_{atom,frame,flags,representation}.csv`). For
each of the 10 MaxVol solutions of step 3, the bulk basis of the DFT-only model's training set is recomputed with the
same seed and call order and traced back to clusters. Clusters 0, 3 and 4 are under-represented in every solution
(0.21–0.26 of their fair share of basis rows; cluster 5 falls below 1 in one solution). Results:

| | 300 K | 2000 K | 5000 K |
|---|---|---|---|
| Frame Spearman: γ_bulk / γ_cluster / switch | 0.20 / 0.21 / 0.21 | 0.17 / 0.23 / 0.23 | 0.08 / 0.47 / 0.47 |
| Atom Spearman: γ_bulk / γ_cluster / switch | 0.55 / 0.53 / 0.51 | 0.40 / 0.40 / 0.37 | 0.15 / 0.27 / 0.18 |

Flags (γ > 1 in ≥ 80% of solutions) in well-represented clusters at 300 / 2000 K: switch = γ_bulk (30 / 55 atoms,
precision 0.40 / 0.42) instead of γ_cluster's 543 / 518 atoms (0.27 / 0.39). In under-represented clusters at 5000 K:
switch = γ_cluster (288 atoms, precision 0.59) instead of γ_bulk's 1. The switch is therefore a flagging rule; as a
continuous atom-level ranking it is slightly worse than either grade alone. max(γ_bulk, γ_cluster) behaves like
γ_cluster.

**Early warning** (`15_early_warning.py`, `slurm/run_early_warning.cmd` → `results/early_warning_*.csv`). The base
model's 40 seeds at 5000 K (13 with a collapse event). Every second saved frame before the event (40 fs apart; 8,249
frames) is re-described with chimes_lsq (two MD-audit frames included as a check: max relative deviation < 5e-17) and
graded with γ_bulk, γ_cluster and the switch rule (5 MaxVol solutions, max over the 64 atoms of a frame).

- Hazard AUROC (frames within Δ ps before an event vs all other pre-event frames and all frames of stable runs,
  bootstrap over runs): 0.49–0.54 for every grade and Δ = 0.2, 0.5, 1, 2 ps; every 95% interval includes 0.5.
- Alarms at the 95th / 99th percentile of stable runs catch 10–11 / 5–6 of 13 events in their last picosecond, but
  fire about 1.2 / 0.25 times per ps in runs that never collapse.

γ therefore does not anticipate the collapse. The stability benefit of §8b comes from the data γ_cluster selects for
the refit, not from detecting instabilities as they develop.

## 9. Caveats

- There is one base model (DFT-only, 42 parameters), one system, one held-out seed and one DFT test set. The
  bootstrap intervals are over frames of the same runs.
- MaxVol variability is included (10 solutions for flagging, 5–10 for active learning). The clustering is k = 6 with
  one seed; `../retrospective/08_cluster_sensitivity.py` shows the 5000 K frame ranking holds for k = 4–8 and three
  seeds (0.30–0.53 vs 0.08 for γ_bulk).
- The active-learning step is a single iteration with ranking fixed against the base set. It is not a multi-cycle
  campaign. The refit models were run in MD (§8b) for 10 ps each; extra MD seeds were run at 5000 K only.
- The switch rule (§8c) has been scored but not used as a selection rule in the active-learning step.
- The t3 models' MD (300 and 2000 K) has γ but no DFT labels. It was run so their flag rates can be compared
  (`flag_summary.csv`).
- Frame t = 0 of every run is a DFT frame and was excluded from labelling and analysis.

## File map and reproduction

Run from the repository root after `source env.sh` (see `../install/`). Retrospective steps 01–02 must have been run
first, because the MD atoms are assigned to the retrospective clusters and graded against `../retrospective/work/A_atomic.npy`.

**Tracked inputs and data** (so every table and figure can be rebuilt without LAMMPS or VASP):

| Path | Content | Size |
|---|---|---|
| `models/<model>/params.txt`, `train_meta.json` | the 5 ChIMES-N models and their exact training frames | 60 kB |
| `data/traj.xyzf.gz`, `data/frame_manifest.csv` | all 3,333 saved Langevin MD frames (33 runs × 101) | 3.2 MB |
| `data/dft_forces.npz` | DFT forces of the 900 labelled frames (eV/Å) | 1.1 MB |
| `results/*.csv` | tables behind Fig. 6 and the calibration table | <100 kB |

**Steps:**

| Step | Script | Output | Needs |
|---|---|---|---|
| 0 | `00_setup_md.py` | `work/md/*/{in.lammps,data.in,params.txt}`, `work/md/jobs.txt` | – |
| MD | `slurm/run_md.cmd` | `work/md/*/traj.lammpstrj` | `LMP_CHIMES` |
| 1 | `01_collect_frames.py` | `data/traj.xyzf.gz`, `data/frame_manifest.csv` | MD output |
| 2 | `02_descriptors.py` | `work/A.npy` | `CHIMES_LSQ`, `MPIRUN` |
| 3 | `03_gamma.py 10` | `work/gamma_md.npz`, `results/flag_summary.csv` | retrospective work/ |
| calib | `dft_calib/make_calib.py`, run VASP in `work/dft_calib/*`, `dft_calib/compare.py` | `results/dft_calibration.csv` | `VASP_GAM`, `VASP_POTCAR_N` |
| 4 | `04_make_dft.py 8` | `work/dft/f<frame>/`, `work/dft/chunk_*.txt` | `VASP_POTCAR_N` |
| DFT | `sbatch --array=0-7 slurm/run_dft_array.cmd` | `work/dft/f*/OUTCAR` | `VASP_GAM` |
| 5 | `05_collect_dft.py` | `data/dft_forces.npz` | OUTCARs |
| 6 | `06_errors.py` | `work/atom_errors_shared_t0_dft_train.csv` | `CHIMES_CALCULATOR` |
| 7 | `07_analysis.py` | `results/audit_shared_t0_dft_train_{atom,frame,flags}.csv` | – |
| 8 | `08_al_step.py` | `results/al_step.csv` | – |
| 9 | `09_figures.py` | `../figures/fig6A_md_flagging.png`, `../figures/fig6B_md_al_step.png` | – |
| 10 | `10_balanced_split.py` | `results/balanced_split.csv`, `results/balanced_split_diagnostics.csv` | step 6 |
| 8b | `08_al_step.py --split balanced --save-coefs` (`slurm/run_al_step.cmd`) | `results/al_step_balanced.csv`, `work/al_models/<split>/coefs.npz` | step 10 |
| 11 | `11_al_md_setup.py balanced 1` | `work/al_md/balanced/<model>__<statepoint>/` | step 8b |
| MD | `sbatch slurm/run_al_md.cmd` | `work/al_md/balanced/*/{log.lammps,traj.lammpstrj}` | `LMP_CHIMES` |
| 12 | `12_al_md_analysis.py balanced` (also `base_seeds`, `balanced_seeds`) | `results/al_md_stability*.csv` | MD |
| 13 | `13_al_md_seed_stats.py` | `results/al_md_seed_stats_{by_k,rules}.csv` | step 12 (all three run sets) |
| 14 | `14_hybrid_md.py` | `results/hybrid_md_{atom,frame,flags,representation}.csv` | steps 3, 6 |
| 15 | `15_early_warning.py prep`, then `slurm/run_early_warning.cmd` (stages descriptors, gamma, analysis) | `work/early_warning/`, `results/early_warning_{frames,hazard,alarm,profile}.csv` | step 12 (`base_seeds`), `CHIMES_LSQ` |

From the tracked data only, run steps 2, 3 and 6–9 (`slurm/run_analysis.cmd`). The MD and DFT steps regenerate the
tracked data. LAMMPS runs with the same seeds are not guaranteed to be bitwise identical across machines or MPI
layouts, so a rerun of the MD gives statistically equivalent, not identical, trajectories.
