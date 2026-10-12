#!/usr/bin/env python3
"""Step 2, arm A of CRITERIA.md: shift-controlled test. For each regime R (>= MIN_FRAMES base frames), fraction f and
draw d: the training set holds a random fraction f of R's frames (at least 2) and 75 % of the frames of every other
regime; the candidates are R's remaining frames. Error model (ridge) and every partition (bulk, element, regime,
cluster) are built from the training set; grades from 5 MaxVol solutions. Active-learning frames (sio2) are not used.
Writes results/shift_<name>.csv: one row per (R, f, draw, partition, element), metrics averaged over the solutions,
bulk-basis representation of R, smallest basis rank and median out-of-span fraction of R's candidate atoms.
usage: 02_shift.py NAME [--cap 32] [--draws 0 1 2] [--fracs 0.02 0.05 0.10 0.25] [--jobs N] [--only REGIME ...] [--nsol 5]
"""
import argparse, os, sys, time, warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

import core as C

MIN_FRAMES = 60
G = {}


def scenario(job):
    R, f, d = job
    warnings.filterwarnings("ignore")
    D, tab, multi, nsol = G["D"], G["tab"], G["multi"], G["nsol"]
    base = tab[~tab.al]
    regimes = sorted(base.regime.unique())
    rng = np.random.default_rng(10_000 + 1000 * d + 10 * regimes.index(R) + int(round(100 * f)))
    tr_frames = []
    for reg, g in base.groupby("regime"):
        fr = g.frame.to_numpy()
        n = max(2, int(round(f * len(fr)))) if reg == R else int(round(0.75 * len(fr)))
        tr_frames.append(rng.choice(fr, n, replace=False))
    tr_frames = np.concatenate(tr_frames)
    train = np.isin(D["frame"], tr_frames)
    cand = (D["regime"] == R) & ~train & np.isin(D["frame"], base.frame)
    t0 = time.time()
    err = C.error_model(D, train, cand)
    parts = C.partitions(D, train, multi)
    rows = []
    rep_R = None
    els = ["all"] + (sorted(np.unique(D["element"])) if multi else [])
    for pname, lab in parts.items():
        Gr, Out, ranks, piv = C.grade_all(D, train, cand, lab, nsol)
        if pname == "bulk":
            rep_R = C.representation(piv, D["regime"], train).get(R, np.nan)
        cand_groups = np.unique(lab[cand])
        for e in els:
            m = np.ones(cand.sum(), bool) if e == "all" else D["element"][cand] == e
            if m.sum() < 30:
                continue
            ok = ~np.isnan(Gr[:, m]).any(0)
            ms = [C.metrics(Gr[t, m][ok], err[m][ok]) for t in range(nsol)]
            rows.append(dict(name=G["name"], regime=R, frac=f, draw=d, partition=pname, element=e, n_cand=int(m.sum()),
                             n_train_frames_R=int(np.isin(tr_frames, base.frame[base.regime == R]).sum()),
                             n_graded=int(ok.sum()), min_rank=min(ranks[int(g)] for g in cand_groups if int(g) in ranks),
                             n_cols=D["n_cols"], outspan_median=float(np.nanmedian(Out[:, m])),
                             **{k: float(np.mean([x[k] for x in ms])) for k in ms[0]}))
    for r in rows:
        r["rep_R"] = rep_R
    print(f"{G['name']} {R} f={f} d={d}: {time.time() - t0:.0f} s", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("--cap", type=int, default=32); ap.add_argument("--draws", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.02, 0.05, 0.10, 0.25]); ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--only", nargs="*"); ap.add_argument("--nsol", type=int, default=C.NSOL); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    D, tab = C.load(a.name, a.cap)
    G.update(D=D, tab=tab, multi=len(np.unique(D["element"])) > 1, nsol=a.nsol, name=a.name)
    base = tab[~tab.al]
    cnt = base.regime.value_counts()
    regimes = [r for r in sorted(cnt.index) if cnt[r] >= MIN_FRAMES and (not a.only or r in a.only)]
    jobs = [(R, f, d) for R in regimes for f in a.fracs for d in a.draws]
    print(f"{a.name}: {len(D['element'])} atoms (cap {a.cap}), {D['n_cols']} nonzero columns, regimes {regimes}, {len(jobs)} scenarios", flush=True)
    if a.jobs > 1:
        with Pool(a.jobs) as p:
            out = p.map(scenario, jobs, chunksize=1)
    else:
        out = [scenario(j) for j in jobs]
    C.RES.mkdir(exist_ok=True)
    df = pd.DataFrame([r for rs in out for r in rs])
    df.to_csv(C.RES / f"shift_{a.name}{a.tag}.csv", index=False)
    pd.set_option("display.width", 250)
    print(df[df.element == "all"].pivot_table(index=["regime", "frac"], columns="partition", values="auroc").round(3).to_string())


if __name__ == "__main__":
    main()
