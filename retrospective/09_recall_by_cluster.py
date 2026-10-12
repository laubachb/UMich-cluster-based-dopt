#!/usr/bin/env python3
"""Step 9: recall of high-error candidates vs labeling budget inside each cluster, per atom and per frame.

Uses the 10 MaxVol solutions saved by 07_raw_and_hybrid.py (work/gamma_retro.npz), so the grades are those of Figs. 2-5.

atom level   candidates = the atoms of cluster c; target = their top-10% force error; score = gamma of the atom
frame level  each candidate frame contributes its atoms of cluster c (frames with < MIN_ATOMS such atoms are skipped);
             frame score = largest gamma among those atoms; frame error = RMS of their per-atom error;
             target = top-10% frames by that error
Scores: gamma_bulk, gamma_cluster; random = budget. Recall at budgets 1-100%, mean and std over the 10 solutions.
Writes results/recall_by_cluster.csv (level, cluster, budget, bulk_mean, bulk_std, cluster_mean, cluster_std, n_items,
n_targets). usage: 09_recall_by_cluster.py
"""
import csv, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.dopt import recall_curve
from common.paths import RETRO_WORK as WORK, RETRO_RESULTS as RES

MIN_ATOMS = 5
BUDGETS = np.concatenate([np.linspace(0.01, 0.2, 20), np.linspace(0.22, 1.0, 20)])


def main():
    man = list(csv.DictReader(open(WORK / "atom_manifest.csv")))
    ch = np.where(np.array([r["kind"] for r in man]) == "chimes")[0]
    e = pd.read_csv(WORK / "atom_errors.csv").set_index("global_atom_idx").loc[ch]
    y, fr = e["mae"].to_numpy(), e["global_frame_idx"].to_numpy()
    lch = np.load(WORK / "labels_candidates.npy")
    g = np.load(WORK / "gamma_retro.npz"); T = g["gb"].shape[0]
    rows = []
    for c in sorted(np.unique(lch)):
        m = lch == c
        # atom level
        info = y[m] > np.quantile(y[m], 0.9)
        curves = {k: np.array([recall_curve(g[k][t][m], info, BUDGETS) for t in range(T)]) for k in ("gb", "gc")}
        for i, b in enumerate(BUDGETS):
            rows.append(dict(level="atom", cluster=c, budget=b, bulk_mean=curves["gb"][:, i].mean(),
                             bulk_std=curves["gb"][:, i].std(), cluster_mean=curves["gc"][:, i].mean(),
                             cluster_std=curves["gc"][:, i].std(), n_items=int(m.sum()), n_targets=int(info.sum())))
        # frame level (cluster-c atoms of each frame)
        d = pd.DataFrame({"f": fr[m], "y": y[m]})
        cnt = d.groupby("f").size(); keep = cnt[cnt >= MIN_ATOMS].index
        ferr = (d.groupby("f").y.apply(lambda v: np.sqrt(np.mean(v ** 2)))).loc[keep]
        finfo = (ferr > ferr.quantile(0.9)).to_numpy()
        fc = {}
        for k in ("gb", "gc"):
            fc[k] = np.array([recall_curve(pd.Series(g[k][t][m]).groupby(fr[m]).max().loc[keep].to_numpy(), finfo, BUDGETS)
                              for t in range(T)])
        for i, b in enumerate(BUDGETS):
            rows.append(dict(level="frame", cluster=c, budget=b, bulk_mean=fc["gb"][:, i].mean(), bulk_std=fc["gb"][:, i].std(),
                             cluster_mean=fc["gc"][:, i].mean(), cluster_std=fc["gc"][:, i].std(), n_items=len(keep),
                             n_targets=int(finfo.sum())))
    df = pd.DataFrame(rows); df.to_csv(RES / "recall_by_cluster.csv", index=False)
    s = df[np.isclose(df.budget, 0.1)]
    print(s[["level", "cluster", "n_items", "n_targets", "bulk_mean", "cluster_mean"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
