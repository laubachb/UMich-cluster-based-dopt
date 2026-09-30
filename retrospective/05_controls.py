#!/usr/bin/env python3
"""Step 5: control arms for Figs. 3-5: which part of 'gamma_cluster + cluster label' carries the signal?
Same data, MaxVol, gradient boosting and GroupKFold-by-frame protocol as 04_gamma_ensemble.py; 5 MaxVol solutions.
Arms: raw scores (no model) and HGB on {bulk}, {cluster}, {ID}, {bulk,ID}, {cluster,ID}, {bulk,cluster,ID}.
Metrics: pooled Spearman / AUROC@q90 (Fig. 3), within-cluster Spearman (Fig. 4), minority-cluster recall@10% (Fig. 5).
Writes results/controls.csv (the control table of the manuscript).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import dopt as E
from common.paths import RETRO_WORK as WORK, RETRO_RESULTS as RES

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
ARMS = {"bulk": ["gb"], "cluster": ["gc"], "id": ["cid"], "bulk_id": ["gb", "cid"],
        "cluster_id": ["gc", "cid"], "bulk_cluster_id": ["gb", "gc", "cid"]}


def oof(X, y, groups, seed):
    X = pd.get_dummies(X, columns=[c for c in X if c == "cid"], dtype=float)
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        p[te] = HistGradientBoostingRegressor(random_state=seed).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
    return p


def main():
    import csv
    man = list(csv.DictReader(open(WORK / "atom_manifest.csv")))
    kind = np.array([r["kind"] for r in man])
    dft_ids, ch_ids = np.where(kind == "dft")[0], np.where(kind == "chimes")[0]
    A = np.load(WORK / "A_atomic.npy")
    A_dft, A_ch = A[E.atom_rows_from_indices(dft_ids)], A[E.atom_rows_from_indices(ch_ids)]
    errs = pd.read_csv(WORK / "atom_errors.csv").set_index("global_atom_idx")
    y = errs.loc[ch_ids, "mae"].to_numpy(float); groups = errs.loc[ch_ids, "global_frame_idx"].to_numpy()
    ldft = np.load(WORK / "labels_dft.npy")
    lch = np.load(WORK / "labels_candidates.npy")
    hid = np.isin(lch, [0, 3, 4]); yh = y[hid]; info = yh > np.quantile(yh, 0.9)
    rows = []
    for t in range(N_TRIALS):
        rng = np.random.default_rng(E.SEED0 + t)
        gb = E.gamma_for_rows(A_ch, E.random_restart_maxvol(A_dft, rng))
        gc = np.full(len(ch_ids), np.nan)
        for c in np.unique(ldft):
            m = np.where(lch == c)[0]
            inv = E.random_restart_maxvol(A[E.atom_rows_from_indices(dft_ids[ldft == c])], rng)
            gc[m] = E.gamma_for_rows(A[E.atom_rows_from_indices(ch_ids[m])], inv)
        feats = pd.DataFrame({"gb": gb, "gc": gc, "cid": lch})
        scores = {"raw_bulk": gb, "raw_cluster": gc}
        for a, cols in ARMS.items():
            scores["hgb_" + a] = oof(feats[cols].copy(), y, groups, E.SEED0 + t)
        for a, s in scores.items():
            r = {"trial": t, "arm": a, "spearman": E.spearman(s, y), "auroc_q90": E.auroc_high_error(s, y),
                 "recall10_hidden": E.recall_curve(s[hid], info, np.array([0.10]))[0]}
            for c in np.unique(lch):
                m = lch == c; r[f"sp_c{c}"] = E.spearman(s[m], y[m])
            rows.append(r)
        print("trial", t, flush=True)
    df = pd.DataFrame(rows); out = RES / "controls.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    print(df.drop(columns="trial").groupby("arm").agg(["mean", "std"]).round(3).T.to_string())


if __name__ == "__main__":
    main()
