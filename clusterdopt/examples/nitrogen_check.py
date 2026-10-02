#!/usr/bin/env python3
"""Check clusterdopt against the manuscript's retrospective study (ChIMES nitrogen).

Needs the retrospective work files (run retrospective/01-04 and 07 first). Fits ClusterDOpt on the DFT atoms
(rows_per_sample=3, 6 clusters, 10 MaxVol solutions), grades the active-learning candidates and compares:
  1. training clusters with retrospective/work/labels_dft.npy, candidate clusters with labels_candidates.npy
  2. gamma_cluster with the manuscript's (work/gamma_retro.npz). The manuscript drew the bulk basis from the same
     random generator before the cluster bases, so its solutions differ from the package's defaults; the check
     redraws them in the manuscript's order to test for exact agreement, and also compares the default solutions
  3. recall of the top-10% error molecular-cluster candidates at a 10% budget (manuscript: 0.43)
usage: python3 clusterdopt/examples/nitrogen_check.py   (from the repository root)
"""
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

from clusterdopt import ClusterDOpt, maxvol

WORK = Path("retrospective/work")


def recall_at(score, info, budget=0.10):
    top = np.argsort(-score)[:int(round(budget * score.size))]
    return info[top].sum() / info.sum()


def main():
    kind = np.array([r["kind"] for r in csv.DictReader(open(WORK / "atom_manifest.csv"))])
    dft, ch = np.where(kind == "dft")[0], np.where(kind == "chimes")[0]
    A = np.load(WORK / "A_atomic.npy")
    rows = lambda idx: (idx[:, None] * 3 + np.arange(3)).ravel()
    A_dft, A_ch = A[rows(dft)], A[rows(ch)]

    t0 = time.time()
    m = ClusterDOpt(n_clusters=6, rows_per_sample=3, n_solutions=10, n_jobs=1).fit(A_dft)
    t_fit = time.time() - t0
    t0 = time.time()
    gc, lab = m.gamma(A_ch, return_clusters=True)
    t_gamma = time.time() - t0
    print(f"fit {t_fit:.0f} s, gamma of {ch.size} candidate atoms {t_gamma:.1f} s")

    ref_dft, ref_ch = np.load(WORK / "labels_dft.npy"), np.load(WORK / "labels_candidates.npy")
    print(f"training clusters identical: {np.array_equal(m.labels_, ref_dft)} "
          f"(ARI {adjusted_rand_score(ref_dft, m.labels_):.4f}); candidate clusters identical: "
          f"{np.array_equal(lab, ref_ch)} (ARI {adjusted_rand_score(ref_ch, lab):.4f})")

    paper = np.load(WORK / "gamma_retro.npz")["gc"]                 # (10, candidates)
    # manuscript order: bulk basis first, then the cluster bases, from one generator per solution
    G = np.empty_like(paper)
    for t in range(paper.shape[0]):
        rng = np.random.default_rng(300 + t)
        maxvol(A_dft, rng)
        for c in range(m.n_clusters_):
            inv = maxvol(A_dft[m._sample_rows(np.where(m.labels_ == c)[0])], rng)[1]
            s = np.where(lab == c)[0]
            G[t, s] = np.abs(A_ch[m._sample_rows(s)] @ inv).max(1).reshape(-1, 3).max(1)
    print(f"manuscript solution order: max |gamma - manuscript gamma| = {np.nanmax(np.abs(G - paper)):.2e}")

    pm = paper.mean(0)
    print(f"default solutions vs manuscript (mean over 10): Spearman {spearmanr(gc, pm)[0]:.3f}, "
          f"gamma > 1: {np.mean(gc > 1):.3%} vs {np.mean(pm > 1):.3%}")

    y = pd.read_csv(WORK / "atom_errors.csv").set_index("global_atom_idx").loc[ch, "mae"].to_numpy(float)
    mol = np.isin(ref_ch, (0, 3, 4)); info = y[mol] > np.quantile(y[mol], 0.9)
    r_pkg = np.mean([recall_at(g[mol], info) for g in m.gamma(A_ch, return_solutions=True)[1]])
    r_paper = np.mean([recall_at(g[mol], info) for g in paper])
    print(f"molecular-cluster recall at 10%: package {r_pkg:.3f}, manuscript {r_paper:.3f}")


if __name__ == "__main__":
    main()
