#!/usr/bin/env python3
"""Step 15: tests T2 (nitrogen) and X (nitrogen) of open_systems/CRITERIA.md on the retrospective candidates.

T: the manuscript clustering (spectral k = 6, 20-NN affinity, seed 42; 15-NN vote for candidates) on the topological
(work/tda.npy) and control (work/ctrl.npy) representations of step 15_tda_features --nitrogen, in place of the force
rows; one MaxVol basis per cluster from the 120 DFT-MD frames, 10 solutions (fresh streams), gamma assigned. Scored as
the SI's k-sweep: Spearman, AUROC and recall at a 10 % budget on the minority candidates of the manuscript clustering
(c0, c3, c4), against the manuscript's gamma_cluster (work/gamma_retro.npz, same metric).
X: k-NN (k = 20) regressors of log error in the standardized 126-feature descriptor: oracle_shift trained on the ALC-2
pool, tested on ALC-3 to ALC-5; oracle_in leave-frame-out on all candidates. Both scored within each manuscript
cluster and on the minority candidates.
Writes results/tda_ceiling_retro.csv and results/tda_ceiling_retro_criteria.csv.
usage: 15_tda_ceiling.py
"""
import csv, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.dopt import SEED0, atom_rows_from_indices as R, random_restart_maxvol, gamma_for_rows, spearman, auroc_high_error, recall_curve
from common.paths import RETRO_WORK as WORK, RETRO_RESULTS as RES, MINORITY_CLUSTERS

N_TRIALS, K = 10, 20
CAL_POOL = "ChIMES.fromALC-1.forALC-2.OUTCAR.xyzf"


def cluster_on(feat_d, feat_c, seed=42):
    sc = StandardScaler().fit(feat_d); Zd, Zc = sc.transform(feat_d), sc.transform(feat_c)
    lab = SpectralClustering(n_clusters=6, affinity="nearest_neighbors", n_neighbors=20, assign_labels="kmeans", random_state=seed, n_jobs=1).fit_predict(Zd)
    return lab, KNeighborsClassifier(15, weights="distance").fit(Zd, lab).predict(Zc)


def main():
    man = list(csv.DictReader(open(WORK / "atom_manifest.csv")))
    kind = np.array([r["kind"] for r in man]); src = np.array([r["source_file"] for r in man]); gf = np.array([int(r["global_frame_idx"]) for r in man])
    dft, ch = np.where(kind == "dft")[0], np.where(kind == "chimes")[0]
    A = np.load(WORK / "A_atomic.npy"); A_ch = A[R(ch)]; X = A.reshape(len(kind), -1)
    y = pd.read_csv(WORK / "atom_errors.csv").set_index("global_atom_idx").loc[ch, "mae"].to_numpy(float)
    lch = np.load(WORK / "labels_candidates.npy"); ldft = np.load(WORK / "labels_dft.npy")
    GC = np.load(WORK / "gamma_retro.npz")["gc"]
    hid = np.isin(lch, MINORITY_CLUSTERS); info = y[hid] > np.quantile(y[hid], 0.9)
    tda, ctrl = np.load(WORK / "tda.npy").astype(float), np.load(WORK / "ctrl.npy").astype(float)
    assert len(tda) == len(kind)

    S = {"gamma_cluster": GC}; comp = {}
    for name, feat in (("tda", tda), ("ctrl", ctrl)):
        ld, lc = cluster_on(feat[dft], feat[ch])
        comp[name] = pd.crosstab(lc, lch)
        Gm = np.full((N_TRIALS, len(ch)), np.nan)
        for t in range(N_TRIALS):
            rng = np.random.default_rng(40_000 + SEED0 + t)
            for c in np.unique(lc):
                m = np.where(lc == c)[0]; dg = dft[ld == c]
                if dg.size < 14:
                    continue
                Gm[t, m] = gamma_for_rows(A_ch[R(m)], random_restart_maxvol(A[R(dg)], rng))
        S[f"gamma_{name}"] = Gm
        print(f"{name}: DFT cluster sizes {np.bincount(ld)}; share of minority candidates in the {name} clusters that hold mostly minority candidates: "
              f"{np.mean(np.isin(lc[hid], np.where(comp[name].loc[:, list(MINORITY_CLUSTERS)].sum(1) > comp[name].sum(1) / 2)[0])):.3f}", flush=True)

    sc = StandardScaler().fit(X[dft]); Zc = sc.transform(X[ch]); ly = np.log(np.maximum(y, 1e-300))
    is_cal = src[ch] == CAL_POOL
    pred = np.full(len(ch), np.nan); pred[~is_cal] = KNeighborsRegressor(K, weights="distance").fit(Zc[is_cal], ly[is_cal]).predict(Zc[~is_cal])
    S["oracle_shift"] = pred[None]
    pin = np.full(len(ch), np.nan); fr = gf[ch]
    for fid in np.unique(fr):
        m = fr == fid; pin[m] = KNeighborsRegressor(K, weights="distance").fit(Zc[~m], ly[~m]).predict(Zc[m])
    S["oracle_in"] = pin[None]

    rows = []
    for s, Gm in S.items():
        for t, v in enumerate(Gm):
            ok = np.isfinite(v)
            r = dict(score=s, trial=t, minority_spearman=spearman(v[hid & ok], y[hid & ok]), minority_auroc=auroc_high_error(v[hid & ok], y[hid & ok]),
                     minority_recall10=recall_curve(v[hid & ok], (y[hid & ok] > np.quantile(y[hid & ok], 0.9)), np.array([0.10]))[0])
            for c in range(6):
                m = (lch == c) & ok; r[f"auroc_c{c}"] = auroc_high_error(v[m], y[m])
            rows.append(r)
    df = pd.DataFrame(rows); df.to_csv(RES / "tda_ceiling_retro.csv", index=False)
    pd.set_option("display.width", 250)
    a = df.groupby("score").mean(numeric_only=True).drop(columns="trial")
    print("\n== minority candidates and within-cluster AUROC (mean over solutions)\n" + a.round(3).to_string())
    for name in comp:
        print(f"\n== {name} clusters (rows) x manuscript clusters (columns), candidates\n" + comp[name].to_string())
    best = a.loc[["gamma_cluster"], "minority_auroc"].max()
    cr = pd.DataFrame([
        dict(criterion="T2-N (tda no loss)", passed=bool(a.loc["gamma_tda", "minority_auroc"] >= a.loc["gamma_cluster", "minority_auroc"] - 0.02),
             detail=f"minority AUROC tda {a.loc['gamma_tda', 'minority_auroc']:.3f}, ctrl {a.loc['gamma_ctrl', 'minority_auroc']:.3f}, cluster {a.loc['gamma_cluster', 'minority_auroc']:.3f}"),
        dict(criterion="X-N (ceiling)", passed=np.nan,
             detail=f"minority AUROC oracle_in {a.loc['oracle_in', 'minority_auroc']:.3f}, oracle_shift {a.loc['oracle_shift', 'minority_auroc']:.3f}; "
                    f"gamma_cluster {best:.3f} (RWV/h_cluster from step 13: 0.96)"),
    ])
    cr.to_csv(RES / "tda_ceiling_retro_criteria.csv", index=False); print("\n== criteria\n" + cr.to_string(index=False))


if __name__ == "__main__":
    main()
