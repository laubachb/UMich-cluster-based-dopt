#!/usr/bin/env python3
"""Step 13: tests H1 and H5 of open_systems/CRITERIA.md (addendum H) on the retrospective candidates.

Reference = the 120 DFT-MD frames (as the gamma bases). Groups = the manuscript's clusters (candidates by its 15-NN
assignment). Regime noise sigma_k = RMS force-component residual of cluster k under the ridge fit (lambda = 1e-8) of the
reference rows (forces in Ha/bohr). Scores (open_systems/core.BayesGrade, information gain of an atom's 3 rows):
IG_hom (one sigma), IG_het (sigma_k), IG_hier (partial pooling, tau by marginal likelihood); gamma_w = MaxVol
gamma_bulk on rows scaled by 1/sigma_k (10 solutions, seeds 300 + t). Baselines: gamma_bulk, gamma_cluster (step 7) and
h_bulk, h_cluster (relative leverage, step 11). Evaluated within each cluster as the manuscript's Fig. 5.
Writes results/info_gain_by_cluster.csv, results/info_gain_criteria.csv.
usage: 13_info_gain.py
"""
import csv, importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.dopt import SEED0, atom_rows_from_indices as R, random_restart_maxvol, gamma_for_rows, spearman, \
    auroc_high_error, recall_curve
from common.paths import RETRO, RETRO_WORK as WORK, RETRO_RESULTS as RES, MINORITY_CLUSTERS
from open_systems.core import BayesGrade, Leverage, group_sigma, fit_hier

N_TRIALS = 10
spec = importlib.util.spec_from_file_location("e3", RETRO / "03_errors.py")
e3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e3)


def main():
    man = list(csv.DictReader(open(WORK / "atom_manifest.csv")))
    kind = np.array([r["kind"] for r in man])
    dft, ch = np.where(kind == "dft")[0], np.where(kind == "chimes")[0]
    A = np.load(WORK / "A_atomic.npy")
    b = e3.forces(list(csv.DictReader(open(WORK / "frame_manifest.csv")))).reshape(-1)
    assert len(b) == A.shape[0]
    Xr, yr, Xc = A[R(dft)], b[R(dft)], A[R(ch)]
    y = pd.read_csv(WORK / "atom_errors.csv").set_index("global_atom_idx").loc[ch, "mae"].to_numpy(float)
    ldft = np.load(WORK / "labels_dft.npy"); lch = np.load(WORK / "labels_candidates.npy")
    g = np.load(WORK / "gamma_retro.npz")
    gr = np.repeat(ldft, 3)

    coef = np.linalg.solve(Xr.T @ Xr + 1e-8 * np.eye(Xr.shape[1]), Xr.T @ yr)
    sigma, s0 = group_sigma(Xr, yr, gr, coef)
    print("regime noise sigma_k (Ha/bohr):", {k: round(v, 5) for k, v in sigma.items()}, "pooled", round(s0, 5), flush=True)
    S = {}
    S["IG_hom"] = BayesGrade(Xr, yr, gr, {k: s0 for k in sigma}, s0).info_gain(Xc, lch)
    S["IG_het"] = BayesGrade(Xr, yr, gr, sigma, s0).info_gain(Xc, lch)
    mh, t, ev = fit_hier(Xr, yr, gr, sigma, s0)
    print(f"IG_hier: t = {t}; log evidence by t: " + ", ".join(f"{k}: {v:.1f}" for k, v in ev.items()), flush=True)
    S["IG_hier"] = mh.info_gain(Xc, lch)
    S["h_bulk"] = Leverage(Xr).grade(Xc)
    hc = np.full(len(ch), np.nan)
    for c in np.unique(lch):
        m = lch == c; hc[m] = Leverage(A[R(dft[ldft == c])]).grade(Xc[R(np.where(m)[0])])
    S["h_cluster"] = hc
    wr, wc = 1 / np.array([sigma[int(k)] for k in gr]), 1 / np.array([sigma[int(k)] for k in np.repeat(lch, 3)])
    GW = np.array([gamma_for_rows(Xc * wc[:, None], random_restart_maxvol(Xr * wr[:, None], np.random.default_rng(SEED0 + t)))
                   for t in range(N_TRIALS)])
    grades = {"gamma_bulk": g["gb"], "gamma_cluster": g["gc"], "gamma_w": GW} | {k: v[None] for k, v in S.items()}

    rows, rec = [], []
    hid = np.isin(lch, MINORITY_CLUSTERS); info = y[hid] > np.quantile(y[hid], 0.9)
    for s, G in grades.items():
        for tr, v in enumerate(G):
            for c in sorted(np.unique(lch)):
                m = lch == c
                rows.append(dict(score=s, trial=tr, cluster=int(c), spearman=spearman(v[m], y[m]), auroc_q90=auroc_high_error(v[m], y[m])))
            rec.append(dict(score=s, trial=tr, recall10_minority=recall_curve(v[hid], info, np.array([0.10]))[0]))
    df, rc = pd.DataFrame(rows), pd.DataFrame(rec)
    df.to_csv(RES / "info_gain_by_cluster.csv", index=False)
    pd.set_option("display.width", 250)
    tab = df.groupby(["cluster", "score"])[["spearman", "auroc_q90"]].mean().unstack("score")
    print("\n== within-cluster metrics (mean over solutions)\n" + tab.round(3).to_string())
    print("\n== minority recall at 10 %\n" + rc.groupby("score").recall10_minority.mean().round(3).to_string())
    a = df.groupby(["cluster", "score"]).auroc_q90.mean().unstack("score").loc[list(MINORITY_CLUSTERS)].mean()
    cr = pd.DataFrame([dict(criterion="H1", passed=bool(a.IG_het >= a.h_bulk + 0.03 and a.IG_hier >= a.h_bulk + 0.03),
                            detail=f"minority mean AUROC IG_het {a.IG_het:.3f}, IG_hier {a.IG_hier:.3f}, h_bulk {a.h_bulk:.3f} "
                                   f"(gamma_cluster {a.gamma_cluster:.3f}, h_cluster {a.h_cluster:.3f}, gamma_w {a.gamma_w:.3f}, "
                                   f"gamma_bulk {a.gamma_bulk:.3f}, IG_hom {a.IG_hom:.3f}); t = {t}")])
    cr.to_csv(RES / "info_gain_criteria.csv", index=False)
    print("\n== criteria\n" + cr.to_string(index=False))


if __name__ == "__main__":
    main()
