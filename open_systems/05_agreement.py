#!/usr/bin/env python3
"""Step 5: the training-only agreement statistic of criterion C7 (CRITERIA.md, addendum of 2026-10-09).

For each arm-A scenario (R, f, draw) of data set NAME, the training set is rebuilt exactly as in 02_shift.py (same
seeds; checked against the candidate counts in results/shift_NAME.csv), the manuscript's clustering is recomputed
(same seed and subsample), and for every cluster g:
  A_g = Spearman(gamma_bulk, gamma_cluster) over the training atoms of g, both grades from bases selected on the
        training rows, mean over 3 MaxVol solutions.
A_R = the mean of A_g over R's candidate atoms (each candidate contributes the A_g of its cluster). No error, force or
grade of a candidate enters A_R.
Writes results/agreement_NAME.csv (one row per scenario).
usage: 05_agreement.py NAME [--jobs N] [--only REGIME ...] [--tag TAG]
"""
import argparse, time, warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import core as C

MIN_FRAMES, NSOL_A = 60, 3
G = {}


def split(R, f, d):
    """identical to the training/candidate split of 02_shift.scenario."""
    D, tab = G["D"], G["tab"]
    base = tab[~tab.al]
    regimes = sorted(base.regime.unique())
    rng = np.random.default_rng(10_000 + 1000 * d + 10 * regimes.index(R) + int(round(100 * f)))
    tr = []
    for reg, g in base.groupby("regime"):
        fr = g.frame.to_numpy()
        n = max(2, int(round(f * len(fr)))) if reg == R else int(round(0.75 * len(fr)))
        tr.append(rng.choice(fr, n, replace=False))
    tr = np.concatenate(tr)
    train = np.isin(D["frame"], tr)
    cand = (D["regime"] == R) & ~train & np.isin(D["frame"], base.frame)
    return train, cand


def scenario(job):
    R, f, d = job
    warnings.filterwarnings("ignore")
    D = G["D"]; t0 = time.time()
    train, cand = split(R, f, d)
    lab = C.partitions(D, train, G["multi"])["cluster"]
    X = D["X"]; tr = np.where(train)[0]
    A = {int(g): [] for g in np.unique(lab[tr])}
    for t in range(NSOL_A):
        rng = np.random.default_rng(C.SEED0 + t)
        gb = C.Basis(X[C.rows_of(tr)], rng).grade(X[C.rows_of(tr)])[0]
        for g in A:
            m = lab[tr] == g
            if m.sum() < 30:
                continue
            idx = tr[m]
            gc = C.Basis(X[C.rows_of(idx)], rng).grade(X[C.rows_of(idx)])[0]
            A[g].append(spearmanr(gb[m], gc)[0])
    Ag = {g: (float(np.mean(v)) if v else np.nan) for g, v in A.items()}
    cand_g = lab[cand]
    vals = np.array([Ag.get(int(g), np.nan) for g in cand_g])
    print(f"{G['name']} {R} f={f} d={d}: A_R {np.nanmean(vals):.3f} ({time.time() - t0:.0f} s)", flush=True)
    return dict(name=G["name"], regime=R, frac=f, draw=d, n_cand=int(cand.sum()), A_R=float(np.nanmean(vals)),
                frac_cand_without_A=float(np.isnan(vals).mean()),
                A_by_cluster=";".join(f"{g}:{v:.3f}:{int((lab[tr] == g).sum())}" for g, v in sorted(Ag.items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("--cap", type=int, default=32); ap.add_argument("--draws", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.02, 0.05, 0.10, 0.25]); ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--only", nargs="*"); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    D, tab = C.load(a.name, a.cap)
    G.update(D=D, tab=tab, multi=len(np.unique(D["element"])) > 1, name=a.name)
    base = tab[~tab.al]; cnt = base.regime.value_counts()
    regimes = [r for r in sorted(cnt.index) if cnt[r] >= MIN_FRAMES and (not a.only or r in a.only)]
    jobs = [(R, f, d) for R in regimes for f in a.fracs for d in a.draws]
    if a.jobs > 1:
        with Pool(a.jobs) as p:
            out = p.map(scenario, jobs, chunksize=1)
    else:
        out = [scenario(j) for j in jobs]
    df = pd.DataFrame(out)
    shift = [f for f in C.RES.glob(f"shift_{a.name}_*.csv")]       # per-job files only (not the merged table)
    if shift:                                          # the rebuilt candidate sets must be those of arm A
        s = pd.concat([pd.read_csv(f) for f in shift])
        s = s[(s.partition == "bulk") & (s.element == "all")].drop_duplicates(["regime", "frac", "draw"]).set_index(["regime", "frac", "draw"]).n_cand
        chk = df.set_index(["regime", "frac", "draw"]).n_cand
        common = chk.index.intersection(s.index)
        assert (chk.loc[common] == s.loc[common]).all(), "candidate sets differ from arm A"
        print(f"candidate counts match arm A for {len(common)} scenarios")
    df.to_csv(C.RES / f"agreement_{a.name}{a.tag}.csv", index=False)
    print(df.drop(columns="A_by_cluster").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
