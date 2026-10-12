#!/usr/bin/env python3
"""Step 25: regime-weighted variance (RWV = IG_het of addendum H, confirmed on mlearn in test V) with groups = the
known state point, on the MD of the DFT-only model (descriptive addendum of open_systems/CRITERIA.md; not a test).

Copy of 23_info_gain.py with the cluster groups replaced by state points: reference atoms (the 90 training frames,
forces in kcal/mol/A) take the state point of their DFT source file (DFT.<sp>.xyzf -> <sp>, as step 20's ref_sp), MD
atoms the state point of their run (df.statepoint). Regime noise sigma_k = RMS force-component residual of state
point k under the DFT-only model's own least-squares fit. Scores: IG_hom (one sigma; rank-equivalent to predictive
variance) and IG_het (open_systems/core.BayesGrade; information gain of an atom's 3 rows, frame score = largest atom
score). gamma_w, IG_hier and the active-learning iteration of step 23 are not computed.
Frame level per state point: Spearman of frame-max score vs frame force RMSE, bootstrap over frames (2,000, seed 0),
paired difference to gamma_cluster; the stored reference scores of step 20 (gamma_bulk, gamma_cluster,
gamma_statepoint, h_bulk, h_cluster, h_statepoint) are scored alongside.
Writes results/info_gain_statepoint_frame.csv.
usage: 25_info_gain_statepoint.py
"""
import csv, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import dopt as E
from common.paths import MD, DATA, RETRO_WORK as RW
from open_systems.core import BayesGrade, group_sigma

MODEL = "shared_t0_dft_train"
N_BOOT = 2000
HA_BOHR_TO_KCAL_A = 627.509474 / 0.529177211


def dft_forces(src, k):
    L = (DATA / src).read_text().splitlines(); i = 0
    for _ in range(k):
        i += int(L[i]) + 2
    n = int(L[i])
    return np.array([l.split()[4:7] for l in L[i + 2:i + 2 + n]], float) * HA_BOHR_TO_KCAL_A


def fit(A, b):
    return np.linalg.lstsq(A, b, rcond=1e-8)[0]


def main():
    A = np.load(RW / "A_atomic.npy")
    man = list(csv.DictReader(open(RW / "atom_manifest.csv")))
    by_frame = {}
    for r in man:
        by_frame.setdefault((r["source_file"], int(r["frame_idx_in_file"])), []).append(int(r["atom_row"]))
    meta = json.load(open(MD / "models" / MODEL / "train_meta.json"))
    train_keys = [(f["source"], i) for f in meta["files"] for i in f["indices"]]
    def block(keys):
        ids = np.concatenate([by_frame[k] for k in keys])
        return A[E.atom_rows_from_indices(ids)], np.concatenate([dft_forces(*k) for k in keys]).ravel()
    A0, b0 = block(train_keys)
    ref_sp = np.concatenate([[k[0][4:-5]] * len(by_frame[k]) for k in train_keys])     # DFT.300K_1.0gcc.xyzf -> 300K_1.0gcc
    c0 = fit(A0, b0)

    fm = pd.read_csv(MD / "data" / "frame_manifest.csv")
    start = np.r_[0, np.cumsum(fm.n_atoms.to_numpy())]
    Am = np.load(MD / "work" / "A.npy", mmap_mode="r")
    df = pd.read_csv(MD / "work" / f"label_partitions_{MODEL}.csv")          # step 20: errors, gamma and h columns
    gidx = start[df.frame.to_numpy()] + df.atom.to_numpy()
    md_sp = df.statepoint.to_numpy()

    # groups = state points (integer codes shared by reference and MD atoms)
    sps = sorted(set(ref_sp) | set(md_sp)); code = {s: i for i, s in enumerate(sps)}
    gr = np.repeat(np.array([code[s] for s in ref_sp]), 3)
    gm = np.array([code[s] for s in md_sp])
    assert set(gm) <= set(gr), "an MD state point has no reference atoms"
    sigma, s0 = group_sigma(A0, b0, gr, c0)
    print("state-point noise sigma_k (kcal/mol/A):", {sps[k]: round(v, 3) for k, v in sigma.items()}, "pooled", round(s0, 3), flush=True)
    models = {"IG_hom": BayesGrade(A0, b0, gr, {k: s0 for k in sigma}, s0), "IG_het": BayesGrade(A0, b0, gr, sigma, s0)}
    Xmd = np.asarray(Am[E.atom_rows_from_indices(gidx)])
    for k, m in models.items():
        df[k] = m.info_gain(Xmd, gm)
    print("scores done", flush=True)

    sc = ["gamma_bulk", "gamma_cluster", "gamma_statepoint", "h_bulk", "h_cluster", "h_statepoint", "IG_hom", "IG_het"]
    frame = []
    for sp, d in df.groupby("statepoint"):
        fr = d.groupby("frame").agg(rmse=("err", lambda e: np.sqrt(np.mean(e ** 2))), **{s: (s, "max") for s in sc})
        B = np.random.default_rng(0).integers(0, len(fr), (N_BOOT, len(fr)))
        boot = {s: np.array([spearmanr(fr[s].to_numpy()[b], fr.rmse.to_numpy()[b])[0] for b in B]) for s in sc}
        for s in sc:
            dd = boot[s] - boot["gamma_cluster"]
            frame.append(dict(statepoint=sp, score=s, groups="statepoint" if s.startswith("IG_") else "stored", n_frames=len(fr),
                              spearman=spearmanr(fr[s], fr.rmse)[0], lo=np.quantile(boot[s], .025), hi=np.quantile(boot[s], .975),
                              minus_cluster=dd.mean(), mc_lo=np.quantile(dd, .025), mc_hi=np.quantile(dd, .975)))
    frm = pd.DataFrame(frame); frm.to_csv(MD / "results" / "info_gain_statepoint_frame.csv", index=False)
    pd.set_option("display.width", 250); print("\n== frame level (IG_* with state-point groups)\n" + frm.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
