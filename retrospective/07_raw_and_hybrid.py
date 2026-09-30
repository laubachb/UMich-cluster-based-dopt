#!/usr/bin/env python3
"""Step 7: raw (model-free) grades, label control and hybrid grades on the retrospective candidates (revised Figs. 3-6).

Uses the 10 MaxVol solutions of 04_gamma_ensemble.py (same seeds and call order, so gamma_bulk and gamma_cluster are
identical to the ones behind Figs. 2-5) and scores every grade without a regressor.

Scores per candidate atom
  raw_bulk      gamma_bulk
  raw_cluster   gamma_cluster
  label         out-of-fold mean error of the atom's cluster (GroupKFold by frame, 5 folds): what the cluster label
                alone knows about error (a regressor on the one-hot label converges to this)
  hyb_max       max(gamma_bulk, gamma_cluster)
  hyb_switch    gamma_cluster for atoms of clusters that are under-represented in the bulk MaxVol basis, gamma_bulk
                otherwise. Representation of cluster c = (share of the 42 bulk basis rows drawn from c's DFT atoms) /
                (share of DFT rows in c); under-represented if < 1. Uses only the training (DFT) data of that solution.
  hyb_rank      max of the two grades' percentile ranks among all candidates (scale-free combination)

Outputs (results/)
  raw_pooled.csv       trial x score: pooled Spearman, AUROC@q90, minority-cluster recall at a 10% budget
  raw_by_cluster.csv   trial x cluster x score: within-cluster Spearman and AUROC@q90 (revised Fig. 5)
  raw_recall.csv       trial x budget: minority-cluster recall curves of every score (revised Fig. 6)
  flag_precision.csv   trial x cluster: atoms flagged (gamma > 1) by gamma_bulk, by gamma_cluster only, by the switch
                       hybrid, and how many of them are in the cluster's own top-10% error (revised Fig. 3)
  bulk_basis_representation.csv   trial x cluster: basis-row share, DFT-row share, ratio
work/gamma_retro.npz   gb[trial, atom], gc[trial, atom] (candidate atoms in manifest order), for 08_cluster_sensitivity.py
usage: 07_raw_and_hybrid.py
"""
import csv
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.dopt import RANK, SEED0, THRESHOLD, atom_rows_from_indices, random_restart_maxvol, gamma_for_rows, \
    spearman, auroc_high_error, recall_curve
from common.paths import RETRO_WORK as WORK, RETRO_RESULTS as RES

N_TRIALS = 10                       # as 04_gamma_ensemble.py
HIDDEN = (0, 3, 4)
BUDGETS = np.concatenate([np.linspace(0.01, 0.2, 20), np.linspace(0.22, 1.0, 20)])


def basis_rows(A_ref, inv):
    """row indices of A_ref forming the MaxVol basis (they map to unit vectors under A_ref @ inv)."""
    B = np.abs(A_ref @ inv)
    rows = np.where((np.abs(B.max(1) - 1) < 1e-8) & (np.abs(B.sum(1) - 1) < 1e-6))[0]
    assert len(rows) == inv.shape[0], (len(rows), inv.shape)
    return rows


def pct_rank(x):
    r = np.empty(len(x)); r[np.argsort(x, kind="stable")] = np.arange(len(x)); return r / (len(x) - 1)


def label_score(lab, y, groups):
    """out-of-fold cluster-mean error."""
    s = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(y, y, groups):
        means = pd.Series(y[tr]).groupby(lab[tr]).mean()
        s[te] = pd.Series(lab[te]).map(means).fillna(y[tr].mean()).to_numpy()
    return s


def main():
    man = list(csv.DictReader(open(WORK / "atom_manifest.csv")))
    kind = np.array([r["kind"] for r in man])
    dft, ch = np.where(kind == "dft")[0], np.where(kind == "chimes")[0]
    A = np.load(WORK / "A_atomic.npy")
    A_dft, A_ch = A[atom_rows_from_indices(dft)], A[atom_rows_from_indices(ch)]
    errs = pd.read_csv(WORK / "atom_errors.csv").set_index("global_atom_idx")
    y = errs.loc[ch, "mae"].to_numpy(float); groups = errs.loc[ch, "global_frame_idx"].to_numpy()
    ldft = np.load(WORK / "labels_dft.npy"); lch = np.load(WORK / "labels_candidates.npy")
    clusters = sorted(int(c) for c in np.unique(ldft))
    hid = np.isin(lch, HIDDEN); info = y[hid] > np.quantile(y[hid], 0.9)
    row_lab = np.repeat(ldft, 3)                        # cluster of every DFT descriptor row
    lab_s = label_score(lch, y, groups)

    pooled, bycl, rec, prec, rep = [], [], [], [], []
    GB, GC = [], []
    for t in range(N_TRIALS):
        rng = np.random.default_rng(SEED0 + t)          # same call order as 04_gamma_ensemble.py
        inv_b = random_restart_maxvol(A_dft, rng)
        gb = gamma_for_rows(A_ch, inv_b)
        gc = np.full(len(ch), np.nan)
        for c in clusters:
            dg, loc = dft[ldft == c], np.where(lch == c)[0]
            if dg.size < RANK or loc.size == 0:
                continue
            inv_c = random_restart_maxvol(A[atom_rows_from_indices(dg)], rng)
            gc[loc] = gamma_for_rows(A[atom_rows_from_indices(ch[loc])], inv_c)
        GB.append(gb); GC.append(gc)

        brow = row_lab[basis_rows(A_dft, inv_b)]
        under = []
        for c in clusters:
            s_basis, s_rows = (brow == c).mean(), (row_lab == c).mean()
            rep.append(dict(trial=t, cluster=c, basis_rows=int((brow == c).sum()), share_basis=s_basis,
                            share_rows=s_rows, ratio=s_basis / s_rows))
            if s_basis / s_rows < 1:
                under.append(c)
        sw = np.where(np.isin(lch, under), gc, gb)
        scores = {"raw_bulk": gb, "raw_cluster": gc, "label": lab_s, "hyb_max": np.maximum(gb, gc),
                  "hyb_switch": sw, "hyb_rank": np.maximum(pct_rank(gb), pct_rank(gc))}

        for s, v in scores.items():
            pooled.append(dict(trial=t, score=s, spearman=spearman(v, y), auroc_q90=auroc_high_error(v, y),
                               recall10_hidden=recall_curve(v[hid], info, np.array([0.10]))[0]))
            for c in clusters:
                m = lch == c
                bycl.append(dict(trial=t, cluster=c, niche="hidden" if c in HIDDEN else "populous", n=int(m.sum()),
                                 score=s, spearman=spearman(v[m], y[m]), auroc_q90=auroc_high_error(v[m], y[m])))
        for b, *r in zip(BUDGETS, *[recall_curve(v[hid], info, BUDGETS) for v in scores.values()]):
            rec.append(dict(trial=t, budget=b, **{f"recall_{s}": x for s, x in zip(scores, r)}))
        for c in clusters:
            m = lch == c; top = y[m] > np.quantile(y[m], 0.9)
            fb, fc_only, fs = gb[m] > THRESHOLD, (gc[m] > THRESHOLD) & (gb[m] <= THRESHOLD), sw[m] > THRESHOLD
            prec.append(dict(trial=t, cluster=c, niche="hidden" if c in HIDDEN else "populous", n=int(m.sum()),
                             n_bulk=int(fb.sum()), top_bulk=int((fb & top).sum()),
                             n_cluster_only=int(fc_only.sum()), top_cluster_only=int((fc_only & top).sum()),
                             n_switch=int(fs.sum()), top_switch=int((fs & top).sum())))
        print(f"trial {t}: under-represented clusters {under}", flush=True)

    np.savez_compressed(WORK / "gamma_retro.npz", gb=np.array(GB), gc=np.array(GC))
    for name, rows in (("raw_pooled", pooled), ("raw_by_cluster", bycl), ("raw_recall", rec),
                       ("flag_precision", prec), ("bulk_basis_representation", rep)):
        pd.DataFrame(rows).to_csv(RES / f"{name}.csv", index=False)
    pd.set_option("display.width", 220)
    P = pd.DataFrame(pooled).groupby("score")[["spearman", "auroc_q90", "recall10_hidden"]].agg(["mean", "std"])
    print(P.round(3).to_string())
    C = pd.DataFrame(bycl).groupby(["cluster", "score"])[["spearman", "auroc_q90"]].mean().unstack("score")
    print(C.round(3).to_string())
    print(pd.DataFrame(rep).groupby("cluster")[["basis_rows", "share_basis", "share_rows", "ratio"]].mean().round(3))


if __name__ == "__main__":
    main()
