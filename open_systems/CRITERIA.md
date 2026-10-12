# Open-data extension: pre-registered design and criteria

Written 2026-10-08, before any Si–O result was computed. The only result seen beforehand is an in-distribution probe on
Si GAP-18 (random 75/25 split within each regime, scratch script, not part of this pipeline): the bulk basis
under-represents the diamond (0.11 of its proportional share), amorphous (0.07), vacancy (0.04) and divacancy (0.00)
regimes, but γ_bulk ranks their error as well as a basis selected within the true regime does. The nitrogen held-out
DFT-MD protocol shows the same thing (manuscript SI, Fig. S4a). The gain of γ_cluster in the manuscript appears only
when the candidates differ from the training data. The tests below therefore build that difference in.

## Data (open, CC-BY or public)

| Name | Source | Elements | Regimes |
|---|---|---|---|
| `si_gap18` | Bartók et al., PRX 8, 041048 (2018); libAtoms/silicon-testing-framework | Si | 12 grouped config types |
| `sio2` | Erhard et al., npj Comput. Mater. (2024), Zenodo 10.5281/zenodo.10419194 (SCAN) | Si, O | SiO₂-composition config types; `*_AL` frames are the candidates of the original active learning |

## Arms (no new DFT)

**A. Shift-controlled test** (both data sets). For each regime R with at least 60 frames and each fraction
f ∈ {0.02, 0.05, 0.10, 0.25}: the training set holds a random fraction f of R's frames (at least 2 frames) and 75 % of
the frames of every other regime; the candidates are the remaining frames of R. Three draws per (R, f).
The error model is a ridge fit (λ = 1e-8) of the ChIMES force rows on the training set; the error of a candidate atom
is its mean absolute force-component deviation. Grades are computed against the training rows, 5 MaxVol solutions.

**B. Active-learning retrospective** (`sio2` only; the analogue of the manuscript's retrospective study). Reference =
the base (non-`_AL`) frames; candidates = the `_AL` frames, grouped by their parent config type. Error model and
grades as in A.

**Partitions**, all built from the training (reference) set only:
1. `bulk`: one basis from all training rows (γ_bulk).
2. `element`: one basis per element (multi-element data only). This is the standard practice for multi-component
   linear potentials and is the fair baseline there.
3. `cluster`: the manuscript's method, applied within each element: concatenated x|y|z rows, standardized, spectral
   clustering (k = 6, 20-nearest-neighbour affinity, k-means labels, seed 42) on at most 15,000 training atoms per
   element, 15-nearest-neighbour distance-weighted vote for all other atoms.
4. `regime`: the true regime labels as the partition (control; separates the effect of the basis from the quality of
   the clustering). Within each element for multi-element data.

**Rank-reduced bases (method extension, needed for multi-element data).** A partition's rows may not span all ChIMES
columns (an O atom has no Si–Si–Si rows). Each basis is therefore selected in the row space of its own training rows:
V_r = right singular vectors with singular value > 1e-8 × the largest; MaxVol on A V_r (r columns); a new row a is
graded as max_j |(a V_r Â_r⁻¹)_j|. When r equals the number of columns this is the manuscript's grade exactly.
The norm of the part of a outside the row space, relative to |a|, is recorded for every graded row.

**Metrics** (within R, or within each candidate group in B): Spearman correlation between grade and error, AUROC for
the 10 % highest-error atoms, recall of those atoms at a 10 % budget. Representation of R in the bulk basis = R's share
of the basis rows divided by its share of the training rows.

## Criteria (fixed before running; Δ = metric of the partition grade − metric of γ_bulk, AUROC unless stated)

- **C1, mechanism.** In each data set, at f = 0.25, at least one regime has a bulk-basis representation below 0.5.
- **C2, shift dependence (regime control).** Over regimes with representation < 0.5 at the given f, the mean Δ of the
  `regime` partition is larger at f = 0.02 than at f = 0.25, in each data set.
- **C3, gain under shift (regime control).** At f ≤ 0.05, the mean Δ of the `regime` partition over regimes with
  representation < 0.5 is positive with a 95 % bootstrap interval (over regimes × draws) that excludes zero.
- **C4, method as published.** C3 with the `cluster` partition.
- **C5, no harm where represented.** In regimes with representation ≥ 1, the mean Δ of `cluster` is above −0.05 at every f.
- **C6, active-learning analogue (`sio2`, arm B).** In candidate groups whose parent regime has representation < 0.5,
  Δ(`cluster`) > 0 and Δ(`regime`) > 0; for multi-element data the comparison is also reported against `element`.

Any criterion that fails is reported as failed. Parameters are not tuned after results are seen; any later change is
listed in the README with its reason.

## Addendum, 2026-10-09 (written after the Si GAP-18 results, before any Si–O result)

Si GAP-18 passed C1–C3 and C5 and failed C4 (the manuscript's clustering, Δ = −0.030 AUROC [−0.039, −0.020]). The
clusters holding the rare-regime candidates were under-represented in the bulk basis (representation 0.15–0.49), so
under-representation alone did not predict a gain. One training-only statistic separated Si from nitrogen:
**A = the within-cluster Spearman correlation between γ_bulk and γ_cluster over the cluster's own training atoms**
(mean over 3 MaxVol solutions). Nitrogen (retrospective reference set): 0.16–0.26 in the minority clusters,
0.34–0.42 in the majority clusters. Si GAP-18 (divacancy, f = 0.05, draw 0): 0.88–0.89 in every large cluster.

**C7 (new, tested on `sio2` only).** For each arm-A scenario, A_R = the candidate-weighted mean of A over the clusters
that hold R's candidate atoms, computed from the training set of that scenario. Prediction: Δ(`cluster`, AUROC) and A_R
are negatively correlated over scenarios (Spearman < 0, 95 % bootstrap interval over regimes × draws excluding 0), and
the mean Δ(`cluster`) of scenarios with A_R < 0.5 exceeds that of scenarios with A_R ≥ 0.8. A is not used by any other
criterion, and C1–C6 are unchanged.

## Addendum, 2026-10-09: third data set `water` (written before any water descriptor, grade or error was computed)

**Data.** Ibrahim, Lysogorskiy, Drautz, Piaggi, "Water phase diagram from a general-purpose atomic cluster expansion
potential" (2026); Zenodo 10.5281/zenodo.18291736, `dataset_water_ACE_rev_PBE_D3.gzip` (CC-BY-4.0; revPBE-D3). Liquid
water and ice Ih, II, III, V, VI and VII; no phase labels. Purpose: a molecular system whose minority regimes differ
from the majority in every molecule, as nitrogen's molecular regime does (Si GAP-18's rare regimes do not).

**Frames.** Periodic, H:O = 2, at least 8 molecules (2,728 of 2,985 frames); at most 600 frames of the tetrahedral
regime (seeded draw), all frames of the other two.

**Regime rule (geometry only; thresholds placed at the edges of the peak and in the gap of the frame histogram).**
c = the mean number of O atoms within 3.3 Å of an O atom (minimum image).
`tetrahedral` c ≤ 4.25 (1,782 frames; ice Ih/Ic, low-pressure liquid), `intermediate` 4.25 < c ≤ 7.5 (472 frames;
ices II, III, V, VI and compressed liquid), `dense` c > 7.5 (471 frames; ice VII/VIII and dense liquid).

**Basis.** As `sio2`: two elements (H, O), 2-body order 12, 3-body order 5, rc_2B = 5.0 Å, rc_3B = 3.5 Å, per-pair
r_min = 0.98 × the smallest distance and Morse λ = the median nearest distance of that pair.

**Tests.** Arm A and criteria C1–C5 and C7 as for `sio2` (three regimes → 36 scenarios). There is no arm B (the data
carry no active-learning history). **Prediction specific to water:** because the regimes differ in every molecule,
the manuscript's clustering should isolate them, so C4 should pass on `water` if γ_bulk is uninformative in an
under-represented regime. If C3 passes and C4 fails on `water`, the atom-level-distinctness explanation offered for
Si GAP-18 is not sufficient.

## Addendum, 2026-10-12: robustness tests R1 (nitrogen, known-condition partitions) and R2 (variance baselines)

Written before any grade, leverage or metric of these tests was computed. The only new fact used is descriptive: every
candidate frame of the nitrogen campaign (ALC-2 to ALC-5 pools) has one of the five densities of the DFT-MD state
points (1.0, 2.0, 2.5, 3.1, 4.5 g cm⁻³), computed from its cell and atom count. Temperature is not stored in the
candidate files, so 300 K and 2000 K candidates (both 1.0 g cm⁻³) cannot be told apart.

### R1. Nitrogen with partitions from known simulation conditions

Question: is the nitrogen gain due to a basis per regime, or to the clustering? Partitions of the reference atoms:
- `density`: 5 groups by frame density (both studies);
- `phase`: 2 groups, 1.0 g cm⁻³ (molecular state points) and the rest (both studies);
- `statepoint`: 6 groups by DFT-MD state point (MD study only, where the temperature of every MD frame is known).
A candidate or MD atom is graded against the basis of its own frame's group. Everything else is the manuscript's
γ_cluster: same MaxVol (common/dopt.py), 10 solutions with seeds 300 + t, references = the 120 DFT-MD frames
(retrospective study) and the 90 training frames (MD study), errors from the existing error files.
Scripts: retrospective/11_label_partitions.py and md_audit/20_label_partitions.py.

Criteria (AUROC for the cluster's 10 % highest-error candidates, mean over solutions; minority clusters c0, c3, c4):
- **R1-1.** AUROC(γ_density) > AUROC(γ_bulk) in each of c0, c3 and c4. Prediction: pass.
- **R1-2.** The density partition recovers at least 75 % of the clustering's gain:
  mean over minority clusters of [AUROC(γ_density) − AUROC(γ_bulk)] ≥ 0.75 × [AUROC(γ_cluster) − AUROC(γ_bulk)].
  Prediction: pass (91–95 % of the minority clusters' DFT atoms are from the 1.0 g cm⁻³ state points).
- **R1-3.** On the 5000 K MD of the DFT-only model, the frame-level Spearman correlation (frame-max grade vs frame force
  RMSE) of γ_statepoint is lower than that of γ_cluster, with a 95 % bootstrap interval of the paired difference
  (resampling frames, 2,000 resamples, seed 0) excluding zero. Prediction: pass, because the high-error 5000 K atoms are
  molecule-like atoms inside a dissociated state point, which an atom-level partition separates and a frame-level
  partition does not. A failure means a frame-level partition is enough for this purpose.
- Descriptive only (no criterion): the same metrics for γ_phase; the 300 K and 2000 K MD; one active-learning
  iteration (top-K by frame-max γ_statepoint and γ_phase, balanced split, K = 2–40, MaxVol streams 11 + draw as in
  md_audit/08). No MD stability runs.

### R2. Linear-model variance baselines

Question: does the gain come from D-optimality, or from grading against a regime-local reference set with any
uncertainty measure? Score: the ridge predictive variance (leverage) of the force fit,
h(a) = Σ_k (a·v_k)² / (s_k² + λ), λ = 1e-8, over the singular triplets of the group's training rows with
s_k > 1e-8 s_1 (the same row space as the rank-reduced γ), divided by the largest h of the group's own training rows
(so that h ≤ 1 inside the group, as for γ), and maximized over an atom's three rows. It is deterministic (no MaxVol).
Arms: nitrogen retrospective (bulk, cluster, density), nitrogen MD (bulk, cluster, statepoint), open data arm A
(bulk, element, regime; cluster on water only, because the clustering is the expensive step and it fails C4 on Si and
SiO₂), SiO₂ arm B (bulk, element, regime, cluster). Script for the open data: open_systems/06_leverage.py; criteria
scored by open_systems/07_robustness.py. Same candidates, errors and bootstrap units as C3.

- **R2-1, bulk variance behaves like γ_bulk.** Nitrogen: AUROC(h_bulk) < 0.6 in each minority cluster. Open data: in
  the C3 scenarios (f ≤ 0.05, representation < 0.5), |mean AUROC(h_bulk) − mean AUROC(γ_bulk)| < 0.05 in each data set.
  Prediction: pass.
- **R2-2, a local reference set helps the variance too.** Open data: in the C3 scenarios, mean
  AUROC(h_regime) − AUROC(h_bulk) > 0 with a 95 % interval excluding zero, in each data set. Nitrogen:
  AUROC(h_cluster) > AUROC(h_bulk) in each minority cluster. Prediction: pass.
- **R2-3, D-optimality vs variance at equal partition.** In the C3 scenarios, |mean Δ(γ_regime) − mean Δ(h_regime)|
  < 0.05 in each data set (Δ relative to the bulk version of the same score), and in nitrogen
  |Δ(γ_cluster) − Δ(h_cluster)| < 0.05 averaged over the minority clusters. Prediction: pass, i.e. the contribution is
  the local reference set, not the D-optimal grade. If local γ beats local h by more than 0.05, the D-optimal grade
  adds something of its own; if local h is better, the manuscript should say so.

As before, failed criteria are reported as failed and no parameter is changed after results are seen.

## Addendum, 2026-10-12 (later): R3, active-learning selection by leverage (nitrogen MD study)

Written after the R1/R2 results (h_bulk ranks within-regime error well on all data sets but ranks 5000 K MD frames
worse than γ_cluster: frame Spearman 0.29 vs 0.47; h_cluster 0.56; h_statepoint 0.70) and before any refit with a
leverage rule. Script: md_audit/21_al_step_leverage.py. Same base model, balanced pool/test split, K = 2, 5, 10, 20,
40, least-squares refits and test sets as md_audit/08 (references: results/al_step_balanced_mv10.csv). Leverage is
computed against the 90 base training frames; a frame's score is the largest leverage of its atoms. Leverage is
deterministic, so each rule is one run (no MaxVol draws).

Rules: `h_bulk` top-K; `h_cluster` top-K (leverage within the atom's spectral cluster, normalized as in R2);
`h_bulk_per_sp` and `h_statepoint_per_sp` (budget split equally among the three state points in the order 5000 K,
2000 K, 300 K, as md_audit/18; reference: results/al_step_per_statepoint.csv).

Reference values (γ_cluster top-K, mean of 10 MaxVol solutions): 5000 K error reduction 0.067 (K = 10), 0.100
(K = 20), 0.116 eV/Å (K = 40); random 0.019 / 0.035 / 0.064; random 5000 K frames 0.046 / 0.085 / 0.143.

- **R3-1.** h_bulk top-K reduces the 5000 K error by less than γ_cluster at K = 10 and K = 20. Prediction: pass
  (its 5000 K frame ranking is weaker, and it is not known which state point it will draw from).
- **R3-2.** h_cluster top-K reduces the 5000 K error by at least 80 % of γ_cluster's reduction at K = 10 and K = 20.
  No directional prediction (its 5000 K frame ranking is better, its choice of state point is unknown).
- **R3-3 (descriptive).** For every rule: state points of the selected frames, 300 K / 2000 K / held-out DFT-MD error.
- Decision rule fixed now: "leverage reproduces the active-learning gain" if a leverage rule meets R3-2's 80 % bar at
  K = 10 and 20 **and** its 300 K error at K = 40 is within 0.02 eV/Å of random selection's (0.206). Only then would MD
  stability runs of the leverage rules be worth their cost.

## Addendum, 2026-10-12 (later): M, the minimum-over-groups rule of Shuang et al. (arXiv 2508.03405), and S, MD stability of leverage selection

### M. Minimum over groups vs assignment to one group

Shuang et al. grade an atom against the active set of every subgroup and keep the minimum. Our γ_cluster assigns the
atom to one cluster (15-NN vote) and grades it against that cluster only. Rule tested: γ_min,P(atom) = min over the
groups g of partition P (same element as the atom) of γ_g(atom), where γ_g uses the same MaxVol bases as the assignment
rule (same seeds, same solutions). If an atom has an out-of-span fraction > 1e-6 against a rank-reduced group basis,
its grade against that group is +∞ (it is not inside that group's volume). Partitions: nitrogen retrospective, cluster
and density; nitrogen MD, cluster and statepoint; open data arm A, regime and cluster (all three data sets, 5 MaxVol
solutions, splits as 02_shift.py). Scripts: retrospective/12_min_rule.py, md_audit/22_min_rule.py,
open_systems/08_min_rule.py, scored by open_systems/09_min_criteria.py.

Prediction and its reason: a molecule-like atom in a hot frame lies inside the volume of a dissociated cluster, so the
minimum drops its grade, and the minimum rule loses the signal the assignment rule finds.
- **M1.** Nitrogen retrospective, minority clusters: mean AUROC(γ_cluster) − mean AUROC(γ_min,cluster) ≥ 0.05.
  Prediction: pass.
- **M2.** Nitrogen 5000 K MD: frame-level Spearman of γ_cluster > γ_min,cluster, paired bootstrap interval (frames,
  2,000 resamples, seed 0) excluding zero. Prediction: pass.
- **M3.** Open data, C3 scenarios: mean [AUROC(γ_regime) − AUROC(γ_min,regime)] > 0 with a 95 % bootstrap interval
  excluding zero, in each data set. Prediction: pass.
- Descriptive: γ_min,cluster on the open data; γ_min,density and γ_min,statepoint on nitrogen; one active-learning
  iteration with top-K by frame-max γ_min,cluster (10 MaxVol solutions).

### S. MD stability of the leverage-selected refits (exploratory)

Deviation, stated before running: the R3 decision rule said MD stability runs of the leverage rules were not worth
their cost (h_cluster raises the 300 K error; h_bulk gains too little at K = 10). They are run anyway because they answer
a separate question asked in the manuscript: does the frame ranking, at fixed state point, lower the 5000 K collapse
rate? Results are reported as exploratory, not as a test.
Models: the h_cluster and h_bulk top-K refits of md_audit/21 (one deterministic model per K = 2, 5, 10, 20, 40), each
run at 5000 K with MD seeds 1–50 (50 runs per K, as the 10 models × 5 seeds of the manuscript; the variation here is
over MD seeds only). Protocol and event definition as md_audit/11 and 12. Comparison: pooled rates of the manuscript
(γ_cluster 0.05, random 5000 K 0.16, random 0.31, γ_bulk 0.45, base 0.33). Expectation, not a criterion: h_cluster,
which selects only 5000 K frames ranked by local novelty, has a pooled rate at or below random 5000 K's 0.16.

## Addendum, 2026-10-12 (later): H, Bayesian information-gain grades with regime noise and partial pooling

Written before any of these grades was computed in the repository. Prior information: a scratch exploration on
2026-10-03 (scripts not kept) reported that a noise-weighted information gain ranked nitrogen error better than
γ_cluster (retrospective minority within-cluster Spearman 0.57–0.62 vs 0.26–0.32; 5000 K MD frame Spearman 0.55 vs
0.47) and that the marginal likelihood preferred a hierarchical model. R2 (this file) showed that plain predictive
variance (h_bulk) already matches the local γ on most tests. Literature checked 2026-10-12: predictive variance for
linear-potential active learning is published (Williams et al., Comput. Mater. Sci. 242, 113074, 2024); per-group
noise in fitting is standard (GAP configuration-type σ); partial pooling of coefficients across regimes was not found.

### Model
Linear force model F = a·c on the training rows of the reference set. Groups k = the partition used for γ_cluster
(nitrogen: the manuscript's clusters with its assignment; open data: element|regime, and the clustering on water only).
- Regime noise σ_k: RMS force-component residual of group k under the pooled fit of the reference set (ridge λ = 1e-8,
  or the DFT-only model's least-squares fit in the MD study). σ_k measures the model error of the 42- or 228-term
  basis in that regime; DFT labels themselves are treated as noise-free.
- `IG_hom`: pooled coefficients, one σ (the pooled residual RMS); rank-equivalent to predictive variance (check vs h_bulk).
- `IG_het`: pooled coefficients, rows weighted by 1/σ_k²; prior precision λ/σ̄² with λ = 1e-8 (reduces to the ridge fit
  when all σ_k are equal).
- `IG_hier`: c_k = c_0 + δ_k, δ_k ~ N(0, τ² I), c_0 with the IG_het prior; τ = t · RMS(c_pooled), t chosen from
  {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1} by the marginal likelihood of the reference forces (σ_k fixed). Posterior
  covariance of c_k = c_0 + δ_k from the joint Gaussian posterior.
- Atom score = information gain of labelling its three force components: ½ log det(I₃ + A₃ C_k A₃ᵀ / σ_k²), C_k the
  posterior covariance of the atom's group coefficients. Frame score = largest atom score.
- `γ_w` (nitrogen only, mechanism check): MaxVol γ_bulk with every reference and candidate row scaled by 1/σ_k
  (10 solutions). If γ_w recovers most of IG_het's gain, the failure of γ_bulk is mainly the equal weighting of regimes;
  if not, it is mainly MaxVol's compression to 42 rows.

### Tests (scripts: retrospective/13_info_gain.py, md_audit/23_info_gain.py, open_systems/10_info_gain.py, scored by open_systems/11_info_criteria.py)
- **H1.** Nitrogen retrospective (reference = 120 DFT-MD frames): mean within-cluster AUROC over the minority clusters of
  IG_het and of IG_hier each ≥ AUROC(h_bulk) + 0.03. Prediction: pass.
- **H2.** Nitrogen 5000 K MD of the DFT-only model (reference = its 90 training frames): frame-level Spearman of IG_het
  > γ_cluster, paired bootstrap interval (frames, 2,000, seed 0) excluding zero. Prediction: pass. IG_hier reported.
- **H3.** Open data, C3 scenarios: mean AUROC(IG_hier, regime groups) − AUROC(h_bulk) > 0 with a 95 % bootstrap interval
  excluding zero. Prediction: pass on water; no prediction on Si GAP-18 and SiO₂.
- **H4.** One active-learning iteration (as md_audit/21): top-K over the whole pool by frame-max IG_het and IG_hier
  (one ranking, no division by state point). Same decision rule as R3: 5000 K error reduction ≥ 80 % of γ_cluster's at
  K = 10 and 20, **and** 300 K error at K = 40 within 0.02 eV/Å of random selection (0.206). No directional
  prediction. A rule that passes is then run for MD stability with the protocol of test S; a rule that fails is not.
  Per-state-point versions are reported as descriptive.
- **H5 (descriptive).** γ_w against γ_bulk and IG_het on the nitrogen tests.

## Addendum, 2026-10-12 (evening): V, confirmation of regime-weighted variance on untouched data (mlearn)

Purpose: decide whether the paper can claim a method. Candidate chosen from the development results above (nitrogen,
Si GAP-18, SiO₂, water), where it was the best or near-best ranker but not uniformly (vs plain variance: nitrogen
+0.11, Si +0.036, water +0.016, SiO₂ −0.036 AUROC). It is tested here unchanged, on data none of the project's
analyses has used: the mlearn single-element datasets (Zuo et al., J. Phys. Chem. A 124, 731, 2020) for Cu, Ge, Li, Mo,
Ni and Si, prepared on 2026-10-04 in cluster-gamma-threshold (42-term ChIMES basis per element, forces eV/Å,
regimes = the datasets' own labels: AIMD at each temperature, vacancy, elastic). Carbon was considered and rejected as
a test because it was used for development in cluster-gamma-threshold.

**Method under test, RWV (regime-weighted variance)** = IG_het of addendum H with groups = the known regime labels:
σ_k = RMS force-component residual of regime k under the ridge fit (λ = 1e-8) of the training rows;
Σ = (Σ_k X_kᵀX_k/σ_k² + 1e-8/σ̄² I)⁻¹; atom score ½ log det(I₃ + A₃ Σ A₃ᵀ/σ_k²) with k = the atom's regime.
**Comparators:** γ_bulk; h_bulk (relative leverage, max over rows); IG_hom (RWV with one σ, isolates the weighting);
γ_regime (one MaxVol basis per regime, assigned); γ_min,regime (minimum over regime bases, Shuang et al.). MaxVol
grades: 5 solutions, seeds 300 + t.
**Design:** arm A unchanged except the minimum regime size: every regime has 20–80 frames, so all regimes with
≥ 20 frames are used (f = 0.02 then keeps the minimum of 2 frames). f ∈ {0.02, 0.05, 0.10, 0.25}, 3 draws, other regimes
75 %, ≤ 32 atoms per frame, error = MAE of the training-set ridge fit, AUROC for the regime's 10 % highest-error atoms.
**Primary scenarios:** f ≤ 0.05 (the representation filter of C3 is dropped, because representation did not predict
where γ_bulk fails). Bootstrap over scenarios, 10,000 resamples, seed 0. Script: open_systems/12_mlearn.py.

- **V1 (primary).** Pooled over the six elements, mean AUROC(RWV) − AUROC(h_bulk) > 0 with a 95 % interval excluding
  zero, **and** the mean difference is positive in at least 4 of the 6 elements.
- **V2.** Pooled mean AUROC(RWV) − AUROC(γ_regime) > 0, interval excluding zero.
- **V3.** Pooled mean AUROC(RWV) − AUROC(γ_min,regime) > 0, interval excluding zero.
- **V4 (sanity).** Pooled mean AUROC(RWV) − AUROC(γ_bulk) > 0, interval excluding zero.
- **Decision, fixed now:** the manuscript claims RWV as a method only if V1 and V2 both pass. Otherwise the paper stays
  an evaluation, and these results are reported as a further test on untouched data.
- Prediction: V4 pass. V1 and V2: no confident prediction; the development evidence is mixed (above).
- Descriptive: all f, per element and per regime; IG_hom vs h_bulk (whether the 3-row information gain itself matters).

## Addendum, 2026-10-12 (night): N, advances on cluster-local D-optimality (scope reopened by the user for this test)

Baselines: γ_cluster (ours: one MaxVol basis per spectral cluster, atom assigned by the 15-NN distance-weighted vote)
and γ_min (Shuang et al., arXiv 2508.03405: minimum over the cluster bases). References reported alongside, not
baselines: γ_bulk, h_bulk, RWV. Literature checked 2026-10-12 (conformal prediction for MLIPs: Hu 2022, Best 2024,
Flexible calibration 2026; mixture-of-experts MLIPs 2026; GMM feature-space UQ, Zhu 2023; QUESTS entropy; locally
weighted ensemble clustering): none combines these with a cluster-local D-optimal grade. Written before any of the
grades below was computed.

### Methods (same bases, seeds and solutions as the baselines; nothing else changes)
- **N-A, soft membership, γ_soft** = exp(Σ_k p_k log γ_k), with p_k the membership probabilities of the same 15-NN
  distance-weighted vote that assigns atoms (so p is one-hot wherever assignment is unambiguous, and γ_soft = γ_cluster
  there) and γ_k the atom's grade against cluster k's basis (+∞ excluded with its weight renormalised; out-of-span rows
  as in addendum M). Descriptive: the arithmetic version Σ_k p_k γ_k. Mechanism: hard assignment loses when clusters
  only partly match regimes (water, Si: γ_min beat γ_cluster), the minimum loses when they do (nitrogen); γ_soft should
  follow the better of the two in each case.
- **N-B, partition ensemble, γ_ens** = exp(mean over 15 clusterings of log γ_cluster), clusterings = k ∈ {4,…,8} ×
  seeds {42, 0, 1} (the manuscript's SI sweep), each with its own assignment; MaxVol streams separate from arm A's.
  Mechanism: one clustering is one noisy draw of the partition; averaging over k and seeds marginalises the choice.
  Not run on SiO₂ (15 clusterings × 2 elements × 228-column bases per scenario is too costly); run on nitrogen, Si,
  water, mlearn.
- **N-C, cluster-conditional conformal calibration, γ_cal** = q_k(x) · γ_cluster(x), where q_k is the Mondrian
  (per-cluster) conformal quantile at level α = 0.1 of |error| / γ_cluster on a calibration set disjoint from the
  training rows and the test candidates (finite-sample quantile ⌈(n_k+1)(1−α)⌉/n_k; clusters with fewer than 20
  calibration atoms use the pooled quantile). Also γ_cal,min for γ_min. Calibration sets: nitrogen retrospective = the
  ALC-2 pool, tested on ALC-3 to ALC-5 (the SI's campaign-order protocol); nitrogen MD = the 30 held-out DFT-MD frames
  (errors of the DFT-only model); open data arm A = the frames of the other regimes not drawn into training (25 % of
  each). Mechanism: γ_cal is a predicted error quantile in force units, so it is comparable across clusters, which
  per-cluster γ is not (H4), and it gives a calibrated flag (the γ > 1 problem).

### Criteria (AUROC for the regime's or cluster's 10 % highest-error atoms unless stated; bootstrap over scenarios,
10,000 resamples, seed 0; C3 scenarios on the open data; nitrogen minority clusters c0, c3, c4)
- **N1 (soft).** Open data (Si, SiO₂, water pooled, cluster partition): mean [AUROC(γ_soft) − AUROC(γ_min)] > 0 and
  mean [AUROC(γ_soft) − AUROC(γ_cluster)] > 0, both intervals excluding zero. Nitrogen: minority AUROC(γ_soft) within
  0.02 of γ_cluster (no loss) and above γ_min. Prediction: pass.
- **N2 (ensemble).** Si and water pooled: mean [AUROC(γ_ens) − AUROC(γ_cluster)] > 0, interval excluding zero.
  Nitrogen: minority AUROC(γ_ens) ≥ AUROC(γ_cluster); 5000 K MD frame Spearman reported. No confident prediction.
- **N3 (calibration).** (a) Cross-cluster ranking: nitrogen retrospective, ALC-3 to ALC-5 candidates pooled over all
  clusters, Spearman(γ_cal, error) ≥ 0.30 (the manuscript's pooled raw γ_cluster: 0.10; label-only regressor: 0.62);
  nitrogen MD, all three state points pooled, Spearman(γ_cal) > Spearman(γ_cluster), bootstrap over frames excluding
  zero. Prediction: pass. (b) Coverage: the 90 % bounds q_k γ cover 85–95 % of candidate errors on the nitrogen MD at
  300 K and 2000 K and in at least 2 of the 3 open data sets (C3 scenarios, mean coverage); 5000 K reported (shifted,
  under-coverage expected). (c) Selection: top-K over the whole pool by frame-max γ_cal meets the H4 decision rule
  (5000 K reduction ≥ 80 % of γ_cluster's at K = 10 and 20, and 300 K error at K = 40 within 0.02 eV/Å of random's
  0.206). No directional prediction; a pass would resolve the open problem of H4.
- **N4 (confirmation on untouched data).** mlearn (Cu, Ge, Li, Mo, Ni, Si; k = 6 clustering as groups; arm A as test V)
  is reserved for ONE method: the one with the largest pooled gain over γ_cluster on the open-data C3 scenarios among
  γ_soft, γ_ens and γ_cal (ranking use). Criterion: pooled mean AUROC gain over γ_cluster and over γ_min both > 0 with
  intervals excluding zero and positive in ≥ 4 of 6 elements. Only a method passing N4 is claimed as an advance.
Scripts: open_systems/13_advance.py (arm A and mlearn), retrospective/14_advance.py, md_audit/24_advance.py,
open_systems/14_advance_criteria.py.

**Deviation, 2026-10-12 (night), recorded before the water and Si runs:** N-B (partition ensemble) is not run on the
open data. Reasons: (1) one Si scenario with 15 clusterings and one MaxVol solution exceeded 24 minutes on a login
node, so the full test (156 scenarios, 5 solutions, two elements for water) does not fit the available allocation;
(2) on the nitrogen retrospective, N2-N's nominal pass (0.836 vs 0.804) is confounded: the ensemble averages 75 grades
per atom before ranking, whereas the baseline metric averages per-solution AUROCs; against the mean-over-10-solutions
γ_cluster (0.858, which equals γ_cal's within-cluster value), the ensemble does not win. N2 is therefore reported as
inconclusive on the retrospective and assessed only on the nitrogen MD (mean grades on both sides: 5000 K frame
Spearman 0.540 vs 0.466). N4's candidate set is reduced to γ_soft and γ_cal.

## Addendum, 2026-10-13: T, a topological representation for the partition; X, the ceiling of within-regime ranking

Written before any topological feature or ceiling regressor was computed. Motivation: the grade must be computed on
the ChIMES force rows (the fit's row space), but the partition can use any per-atom representation. The force-row
clustering is not rotation invariant and does not isolate the atoms around defects in network solids (Si, SiO₂), where
it lost to the true regime labels (C4). A topological descriptor of each atom's neighbourhood is rotation invariant and
sensitive to rings and cavities.

### T. Partition on a persistent-homology descriptor
- **Representation `tda`** (open_systems/15_tda_features.py): for every atom, the point cloud of its neighbours within
  the 3-body cutoff rc3 of the data set (at least the 12 nearest if fewer), periodic images included, centred on the
  atom. Vietoris–Rips persistence (ripser, maxdim 1, threshold rc3). Features: the 12 smallest H0 death times (padded
  with rc3); H1 count, total persistence, longest persistence, mean birth, mean death; a 4 × 4 persistence image of H1
  over birth ∈ [0, rc3], persistence ∈ [0, rc3/2] (Gaussian width 0.2 Å). 33 features, rotation and permutation invariant.
- **Control `ctrl`**: the 12 smallest neighbour distances (padded with rc3), an 8-bin histogram of the angles between
  neighbour pairs, and the neighbour count. 21 features, also rotation invariant but with no topology beyond distances
  and angles. It separates the effect of rotation invariance from that of topology.
- Clustering, assignment, bases, grading, scenarios and metrics exactly as the manuscript clustering (spectral k = 6,
  20-NN affinity, 15-NN vote), with the representation swapped. Partitions compared per scenario: cluster (force rows,
  arm A), tda, ctrl, regime (true labels). Script: open_systems/16_tda_partition.py; nitrogen:
  retrospective/15_tda_ceiling.py (bases from the 120 DFT-MD frames, 10 solutions, scored on the minority candidates of
  the manuscript clustering as the SI's k-sweep does).
- **T1.** Si and SiO₂ pooled, C3 scenarios: mean [AUROC(γ_tda) − AUROC(γ_cluster)] ≥ 0.05 with a 95 % interval excluding
  zero, and ≥ 0.5 × mean [AUROC(γ_regime) − AUROC(γ_cluster)]. No confident prediction.
- **T2.** Water (C3 scenarios) and nitrogen (minority candidates): mean AUROC(γ_tda) ≥ AUROC(γ_cluster) − 0.02 (no loss
  where the force-row clustering already works). Prediction: pass.
- **T3 (attribution, only if T1 passes).** Si and SiO₂ pooled: mean [AUROC(γ_tda) − AUROC(γ_ctrl)] > 0, interval
  excluding zero; otherwise the gain is rotation invariance, not topology.

### X. Ceiling of descriptor-based within-regime ranking (descriptive, with a decision rule)
Per arm-A scenario, two k-nearest-neighbour error regressors (k = 20, distance-weighted, standardized 126-feature
force-row descriptor, target log error), neither of which a grade could use:
- `oracle_shift`: trained on the calibration atoms (other regimes not drawn into training, errors known), predicting
  the candidates: the ceiling of a supervised error model under the same shift the grades face.
- `oracle_in`: leave-frame-out within the candidates of R (trained on R's other candidate frames): the ceiling of any
  map from this descriptor to error when labels of the regime are available.
Nitrogen: `oracle_shift` trained on the ALC-2 pool, tested on ALC-3 to ALC-5; `oracle_in` leave-frame-out on the
candidates. Script: open_systems/18_ceiling.py (--score joins the best existing grade per scenario: max over
γ_regime, h_regime, RWV, γ_cal where available).
- **Decision, fixed now:** headroom = mean AUROC(oracle_in) − mean AUROC(best grade) over the C3 scenarios of a data
  set. If headroom < 0.05, within-regime ranking is treated as at ceiling for that data set and no further
  within-regime method work is done on it; if ≥ 0.10, a supervised error model (addendum N's calibration extended with
  local residual features) is the next test. oracle_shift ≤ best grade would mean that even a supervised model cannot
  beat the grades under shift.

## Addendum (descriptive), applying RWV with state-point groups on the nitrogen MD

Written 2026-10-12 before the computation, after test V confirmed RWV (regime-weighted variance, IG_het with known
regime labels) on mlearn. The nitrogen MD tests H2/H4 ran IG_het with the spectral clusters as groups (5000 K frame
Spearman 0.508 vs γ_cluster 0.466, γ_statepoint 0.703, h_statepoint 0.697). Here IG_het and IG_hom are computed
unchanged (script md_audit/25_info_gain_statepoint.py, a copy of md_audit/23 without γ_w, IG_hier and the
active-learning iteration) with groups = the known state point of every atom: reference atoms (the 90 training frames)
take the state point of their DFT source file, MD atoms that of their run; σ_k = RMS force-component residual of state
point k under the DFT-only model's least-squares fit. Reported: frame-level Spearman per state point with the bootstrap
over frames (2,000, seed 0) and the paired difference to γ_cluster, next to the stored γ_statepoint and h_statepoint
(results/info_gain_statepoint_frame.csv). This is descriptive: it shows what the confirmed method gives on the nitrogen
MD when the regime labels are known, as on mlearn; no criterion is attached and nothing is claimed from it.

## Addendum (exploratory), MD stability of soft-membership selection

Written 2026-10-12 before any run, after test N: γ_soft was the only rule to meet the H4 selection decision rule
(md_audit/results/al_step_advance.csv, advance_md_criteria.csv: 5000 K reduction 0.060 / 0.088 at K = 10 / 20 vs the
0.053 / 0.080 bars; 300 K at K = 40 0.206 vs random 0.206), and addendum H says a rule that passes is run for MD
stability with the protocol of test S. Models: the γ_soft top-K refits of MaxVol draw 0 of md_audit/24 (one model per
K = 2, 5, 10, 20, 40, reproduced by md_audit/26_soft_coefs.py, checked against the draw-0 rows of al_step_advance.csv),
each run at 5000 K, 2.0 g/cm³ with MD seeds 1–50 (50 runs per K; variation over MD seeds only, as test S). Protocol
and event definition as md_audit/11 and 12 (work/al_md/soft_seeds, results/al_md_stability_soft_seeds.csv). Reported:
pooled 5000 K event rate over K with a Wilson 95 % interval and the K ≤ 10 rate, next to γ_cluster (manuscript: 10/225
pooled, 3/135 at K ≤ 10), h_cluster (test S: 18/250, 17/150) and random 5000 K frames (39/250, 31/150). Expectation,
not a criterion: a pooled rate at or below h_cluster's 0.072 (at K ≤ 10, draw 0 of γ_soft selects only 5000 K frames,
as h_cluster does). Exploratory, like test S: not a pre-registered test.
